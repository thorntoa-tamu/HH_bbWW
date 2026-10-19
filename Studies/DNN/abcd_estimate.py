"""ABCD background estimate and signal significance for the ABCDisCo DNN.

Builds the ABCD plane from a single trained DisCo model:
  axis 1 = DNN score, cut at fc
  axis 2 = HME in the signal on-mass window (in) vs sideband (out)

Regions (signal region = A):
  A: score >= fc  and  HME in window
  B: score >= fc  and  HME sideband
  C: score <  fc  and  HME in window
  D: score <  fc  and  HME sideband

Data-driven background in A:  N_A^pred = N_B * N_C / N_D  (background only).
Signal significance uses the Asimov formula, stat-only and with the ABCD
prediction uncertainty folded in.

All yields are physics-weighted (weight_Central); signal is scaled by the
bbWW branching factor. Reuses the exact loader/inference from
plot_dnn_hme_validation.py so the scores match the validation plots.

Skips closure / signal-contamination r by design: for HH the signal is tiny,
so the sidebands (B, C, D) are background-dominated and r is negligible.
"""

import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as skm

import plot_dnn_hme_validation as pdv
from plot_dnn_hme_validation import load_data, run_model, VALIDATION_FILE, SIGNAL_MASS


def set_model(model_dir):
    """Point the shared loader/inference at a given trained model dir."""
    pdv.MODEL_DIR = model_dir
    pdv.MODEL_PATH = os.path.join(model_dir, "val_combined_loss.onnx")
    pdv.CONFIG_PATH = os.path.join(model_dir, "dnn_config.yaml")
    return pdv.MODEL_PATH

# Branching factor applied to signal (matches DNN_Class_HistTuples.py). Converts
# the stored signal normalization to the bbWW final state.
BR_FACTOR = 0.0264215349425664

# The validation file is one parity split of NPARITY. Scale yields to the full
# dataset (matches the nParity=4 convention in the training FOM metric). Set to 1
# to quote yields for this file only, or run over all merged parities instead.
NPARITY = 4

# Require at least this many raw MC events in each sideband region for the ABCD
# prediction to be considered stable.
MIN_RAW_EVENTS = 20

# Default model: lambda=100 (same as plot_dnn_hme_validation.py). Override with
# --model-dir to run a different lambda.
DEFAULT_MODEL_DIR = (
    "/eos/user/t/thorntoa/HH_bbWW/DNNTraining/m600_disco100/Run3_2022/"
    "DNN_DoubleLepton_Resolved_Disco_Training0_par0_m600/"
    "DNN_DoubleLepton_Resolved_Disco_Training0_par0_m600"
)


def asimov_z(s, b, sigma_b=0.0):
    """Asimov discovery significance. sigma_b > 0 folds in a background
    uncertainty (the ABCD prediction error)."""
    if b <= 0 or s <= 0:
        return 0.0
    if sigma_b <= 0:
        return np.sqrt(max(2.0 * ((s + b) * np.log(1.0 + s / b) - s), 0.0))
    s2 = sigma_b * sigma_b
    t1 = (s + b) * np.log((s + b) * (b + s2) / (b * b + (s + b) * s2))
    t2 = (b * b / s2) * np.log(1.0 + s2 * s / (b * (b + s2)))
    return np.sqrt(max(2.0 * (t1 - t2), 0.0))


def region_yield(w, mask):
    """Weighted yield and squared-weight sum (for the stat error) in a region."""
    ww = w[mask]
    return ww.sum(), np.square(ww).sum(), int(mask.sum())


def _dcorr2_once(x, y, w):
    """Weighted squared distance correlation on one sample (V-statistic form,
    same estimator the DisCo loss minimizes). dCorr2 = 0 iff independent."""
    w = w / w.sum()
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    # Weighted double centering
    a_i = a @ w
    b_i = b @ w
    a_mean = a_i @ w
    b_mean = b_i @ w
    A = a - a_i[:, None] - a_i[None, :] + a_mean
    B = b - b_i[:, None] - b_i[None, :] + b_mean
    wij = np.outer(w, w)
    dcov = np.sum(wij * A * B)
    dvx = np.sum(wij * A * A)
    dvy = np.sum(wij * B * B)
    denom = np.sqrt(dvx * dvy)
    return dcov / denom if denom > 0 else np.nan


def dcorr2(x, y, w, n_sub=2000, n_rep=5, seed=0):
    """dCorr2 averaged over n_rep random subsamples (it is O(n_sub^2), so it
    cannot run on the full sample). Returns the mean over draws."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n <= n_sub:
        return _dcorr2_once(x, y, w)
    vals = []
    for _ in range(n_rep):
        idx = rng.choice(n, n_sub, replace=False)
        vals.append(_dcorr2_once(x[idx], y[idx], w[idx]))
    return float(np.nanmean(vals))


def roc_weighted(y, score, w):
    """Weighted ROC (fpr, tpr, AUC). Negative NLO weights are clipped to 0 so
    the weighted FPR stays monotonic (a small fraction of events)."""
    wc = np.clip(w, 0, None)
    fpr, tpr, _ = skm.roc_curve(y, score, sample_weight=wc)
    auc = skm.roc_auc_score(y, score, sample_weight=wc)
    return fpr, tpr, auc


def disco_tag(model_dir):
    """Derive a short label like 'disco10' from a model dir path, falling back
    to '<production>_<TrainingN>' (e.g. prod_TTonly_Training1)."""
    for part in model_dir.split(os.sep):
        if "disco" in part:
            return part
    return pdv.model_label(model_dir)


def process_closure(mask, score, w, in_window, ok, fc):
    """All-region ABCD closure (pred/true) for one process at score cut fc."""
    hi = score >= fc
    sb = ok & ~in_window
    yB = w[mask & hi & sb].sum()
    yC = w[mask & ~hi & in_window].sum()
    yD = w[mask & ~hi & sb].sum()
    yA = w[mask & hi & in_window].sum()
    if min(yB, yC, yD, yA) <= 0:
        return np.nan
    return (yB * yC / yD) / yA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Trained model dir containing val_combined_loss.onnx + dnn_config.yaml",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Output subdir tag (default: derived from the model dir name)",
    )
    parser.add_argument(
        "--compare", default="",
        help="Comma-separated extra model dirs to overlay on the ROC (same "
             "feature set assumed, e.g. the disco0/disco10 siblings).",
    )
    parser.add_argument(
        "--bkg", default="all", choices=["all", "TT"],
        help="Background used for the ABCD yields, Z and ROC: all (class_value "
             "> 0) or TT only (class_value == 1).",
    )
    args = parser.parse_args()

    model_path = set_model(args.model_dir)
    print(f"Model: {model_path}")

    tag = args.tag or disco_tag(args.model_dir)
    if args.bkg != "all" and not args.tag:
        tag = f"{tag}_vs{args.bkg}"
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"abcd_estimate_m{SIGNAL_MASS}_{tag}"
    )
    os.makedirs(output_dir, exist_ok=True)

    features, data = load_data()
    print("Running inference")
    score = run_model(features)

    cv = data["class_value"]
    xm = data["X_mass"]
    w = data["weight_Central"].astype("float64")
    hme = data["DeepHME_mass"].astype("float64")
    hme_ok = np.isfinite(hme) & (hme > 0)

    sig_mask = (cv == 0) & (xm == SIGNAL_MASS) & hme_ok
    bkg_sel = (cv == 1) if args.bkg == "TT" else (cv > 0)
    bkg_mask = bkg_sel & hme_ok

    # On-mass window = signal weighted central 68% of HME (same as the plots)
    sig_hme = hme[sig_mask]
    sig_w = w[sig_mask]
    order = np.argsort(sig_hme)
    cdf = np.cumsum(sig_w[order]) / sig_w.sum()
    lo = sig_hme[order][np.searchsorted(cdf, 0.16)]
    hi = sig_hme[order][np.searchsorted(cdf, 0.84)]
    lo, hi = round(lo / 10) * 10, round(hi / 10) * 10
    print(f"HME on-mass window (signal central 68%): [{lo}, {hi}] GeV")

    in_window = hme_ok & (hme >= lo) & (hme < hi)
    sideband = hme_ok & ~in_window

    sig_w_scaled = w * BR_FACTOR  # applied to signal yields only

    # Scan the DNN score cut. Grid spans the bulk of the signal score range.
    fc_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.9, 46),
                np.linspace(0.9, 0.999, 50),
            ]
        )
    )

    rows = []
    for fc in fc_grid:
        hi_f = score >= fc

        # Background yields per region
        A = bkg_mask & hi_f & in_window
        B = bkg_mask & hi_f & sideband
        C = bkg_mask & ~hi_f & in_window
        D = bkg_mask & ~hi_f & sideband

        yB, w2B, nB = region_yield(w, B)
        yC, w2C, nC = region_yield(w, C)
        yD, w2D, nD = region_yield(w, D)
        yA_true, _, nA = region_yield(w, A)

        if yD <= 0 or yB <= 0 or yC <= 0:
            continue

        raw_pred = yB * yC / yD
        b_pred = NPARITY * raw_pred
        b_true = NPARITY * yA_true

        # Relative stat error of B*C/D from the three sidebands (single parity),
        # reduced by sqrt(NPARITY) for the full-dataset statistics.
        relerr2 = w2B / (yB * yB) + w2C / (yC * yC) + w2D / (yD * yD)
        sigma_b = b_pred * np.sqrt(relerr2 / NPARITY)

        # Signal yield in the signal region A
        s = NPARITY * sig_w_scaled[sig_mask & hi_f & in_window].sum()

        z_stat = asimov_z(s, b_pred)
        z_syst = asimov_z(s, b_pred, sigma_b)

        stable = (nB >= MIN_RAW_EVENTS) and (nC >= MIN_RAW_EVENTS) and (
            nD >= MIN_RAW_EVENTS
        )

        rows.append(
            dict(
                fc=fc,
                S=s,
                B_pred=b_pred,
                B_pred_err=sigma_b,
                B_true=b_true,
                closure=(b_pred / b_true) if b_true > 0 else np.nan,
                Z_stat=z_stat,
                Z_syst=z_syst,
                nA=nA,
                nB=nB,
                nC=nC,
                nD=nD,
                stable=stable,
            )
        )

    # Pre-selection baseline (no score cut): whole in-window region
    s0 = NPARITY * sig_w_scaled[sig_mask & in_window].sum()
    b0 = NPARITY * w[bkg_mask & in_window].sum()
    z0 = asimov_z(s0, b0)
    print(
        f"Pre-selection (window, no score cut): S={s0:.3f}, B={b0:.1f}, "
        f"Z_asimov={z0:.4f}"
    )

    # Best stable working point by stat-only Z
    stable_rows = [r for r in rows if r["stable"]]
    pool = stable_rows if stable_rows else rows
    best = max(pool, key=lambda r: r["Z_stat"])
    print("\nBest working point (max Z_stat among stable cuts):")
    print(
        f"  fc      = {best['fc']:.4f}\n"
        f"  S       = {best['S']:.4f}\n"
        f"  B_pred  = {best['B_pred']:.3f} +/- {best['B_pred_err']:.3f}"
        f"  (B_true = {best['B_true']:.3f}, closure = {best['closure']:.3f})\n"
        f"  Z_stat  = {best['Z_stat']:.4f}   Z_syst = {best['Z_syst']:.4f}\n"
        f"  raw evts A/B/C/D = {best['nA']}/{best['nB']}/{best['nC']}/{best['nD']}"
    )
    print(f"  Enhancement over pre-selection: Z_stat/Z0 = {best['Z_stat']/z0:.2f}x")

    # Plot: predicted vs true background, and significance, vs fc
    fcv = np.array([r["fc"] for r in rows])
    bpred = np.array([r["B_pred"] for r in rows])
    bperr = np.array([r["B_pred_err"] for r in rows])
    btrue = np.array([r["B_true"] for r in rows])
    zstat = np.array([r["Z_stat"] for r in rows])
    zsyst = np.array([r["Z_syst"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)

    ax1.fill_between(
        fcv, bpred - bperr, bpred + bperr, color="#2a78d6", alpha=0.2,
        label="ABCD pred +/- stat",
    )
    ax1.plot(fcv, bpred, color="#2a78d6", label="ABCD predicted bkg in A")
    ax1.plot(fcv, btrue, color="#6e6d66", linestyle="--", label="True MC bkg in A")
    ax1.set_yscale("log")
    ax1.set_ylabel("Background yield in A")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(
        f"m{SIGNAL_MASS} ABCD, window [{lo}, {hi}] GeV (x{NPARITY} parity scaling)"
    )

    ax2.plot(fcv, zstat, color="#e34948", label="Z Asimov (stat only)")
    ax2.plot(fcv, zsyst, color="#eda100", label="Z Asimov (+ ABCD unc)")
    ax2.axhline(z0, color="#6e6d66", linestyle=":", label=f"pre-selection Z={z0:.2f}")
    ax2.axvline(best["fc"], color="#1baf7a", linestyle=":", label=f"best fc={best['fc']:.3f}")
    ax2.set_xlabel("DNN score cut fc")
    ax2.set_ylabel("Significance Z")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"abcd_significance_m{SIGNAL_MASS}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Decorrelation diagnostics: per-process dCorr2(score, HME) and closure.
    # The all-bkg number is TT-dominated and hides a per-process failure (DY),
    # so break it out by process. dCorr2 is the quantity the DisCo loss uses;
    # Pearson (as in validate/summary.txt) only sees linear correlation.
    # ------------------------------------------------------------------
    processes = {
        "TT": (cv == 1) & hme_ok,
        "DY": (cv == 2) & hme_ok,
        "Other": (cv == 3) & hme_ok,
        "All bkg": bkg_mask,
    }
    # Percentile-matched score cuts (each process cut at its own score quantile,
    # so closure is not distorted by a fixed threshold landing in a sparse tail).
    pcts = [0.5, 0.8, 0.9, 0.95]

    print("\nDecorrelation diagnostics (background):")
    header = f"  {'process':8s} {'sum(w)':>10s} {'dCorr2':>8s}  " + " ".join(
        f"clos@{int(p*100)}" for p in pcts
    )
    print(header)
    for name, m in processes.items():
        d2 = dcorr2(score[m], hme[m], w[m])
        clos = [process_closure(m, score, w, in_window, hme_ok,
                                np.quantile(score[m], p)) for p in pcts]
        print(
            f"  {name:8s} {w[m].sum():10.1f} {d2:8.4f}  "
            + " ".join(f"{c:7.3f}" for c in clos)
        )
    # ------------------------------------------------------------------
    # ROC: signal(m{mass}) vs all background, weighted. The main-model score is
    # reused; any --compare model dirs are run on the same features and overlaid
    # (assumes an identical feature list, e.g. the disco0/10/100 siblings). Shows
    # the classifier-quality cost of the DisCo penalty that closure alone hides.
    # ------------------------------------------------------------------
    roc_sig = (cv == 0) & (xm == SIGNAL_MASS)
    roc_bkg = bkg_sel
    sel = roc_sig | roc_bkg
    y_roc = roc_sig[sel].astype(int)

    roc_models = [(tag, score)]
    compare_dirs = [d for d in args.compare.split(",") if d.strip()]
    for cdir in compare_dirs:
        set_model(cdir)
        try:
            csc = run_model(features)
        except Exception as e:
            print(f"  ROC compare skipped for {cdir}: {e}")
            continue
        roc_models.append((disco_tag(cdir), csc))
    set_model(args.model_dir)  # restore main model

    bkg_label = "TT" if args.bkg == "TT" else "all background"
    print(f"\nROC (signal vs {bkg_label}, weighted):")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, sc in roc_models:
        fpr, tpr, auc = roc_weighted(y_roc, sc[sel], w[sel])
        ax.plot(tpr, 1 - fpr, label=f"{name}  AUC={auc:.4f}")
        print(f"  {name:12s} AUC={auc:.4f}")
    ax.plot([0, 1], [1, 0], "k:", lw=0.8)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection (1 - fpr)")
    ax.set_title(f"m{SIGNAL_MASS} ROC vs {bkg_label} (weighted)")
    ax.legend(frameon=False)
    fig.tight_layout()
    roc_path = os.path.join(output_dir, f"roc_m{SIGNAL_MASS}.pdf")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {os.path.join(output_dir, f'abcd_significance_m{SIGNAL_MASS}.pdf')}")
    print(f"Wrote {roc_path}")


if __name__ == "__main__":
    main()
