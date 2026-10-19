"""Shape validation plots for the ABCDisCo DNN (DNN_Class_HistTuples pipeline).

Runs a trained m600 DisCo model on the held-out parity file and makes:
  1. DNN score shapes for signal (m600) and each background
  2. HME shape in DNN score slices per background (ABCD sculpting check)
  3. Signal vs background DNN score shape for events inside the HME window
  4. Cross-section-scaled score in the HME window with S/sqrt(B) vs score cut

Plots 1-3 are weighted by weight_Central and normalized to unit area. Plot 4
keeps the physics normalization (signal x BR_FACTOR, x NPARITY) to pick the
initial ABCD score cut.

"""

import os
import re
import glob
import argparse
import numpy as np
import uproot
import yaml
import onnxruntime as ort
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = (
    "/eos/user/t/thorntoa/HH_bbWW/DNNTraining/m600_disco100/Run3_2022/"
    "DNN_DoubleLepton_Resolved_Disco_Training0_par0_m600/"
    "DNN_DoubleLepton_Resolved_Disco_Training0_par0_m600"
)
MODEL_PATH = os.path.join(MODEL_DIR, "val_combined_loss.onnx")
CONFIG_PATH = os.path.join(MODEL_DIR, "dnn_config.yaml")
# par0 model: trained on parity0, tested on parity1 -> parity2 is unseen
VALIDATION_FILE = (
    "/eos/user/t/thorntoa/DNNDatasets/ResolvedDataset_v2605a_Disco/Dataset/"
    "nParity2_Merged.root"
)
SIGNAL_MASS = 600
# Same values as abcd_estimate.py: bbWW branching factor for signal, and the
# validation file is one of NPARITY splits, so scale yields to the full dataset.
BR_FACTOR = 0.0264215349425664
NPARITY = 4
# Minimum raw MC background events above a cut for its S/sqrt(B) to count
MIN_RAW_EVENTS = 20
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_plots_m600_disco100")

# Categorical palette (validated, CVD-safe)
COLORS = {
    "Signal": "#e34948",
    "TT": "#2a78d6",
    "DY": "#eda100",
    "Other": "#1baf7a",
    "All bkg": "#6e6d66",
}
# Ordinal blue ramp for ordered slices (low -> high)
SCORE_SLICE_COLORS = ["#b7d3f6", "#6da7ec", "#2a78d6", "#0d366b"]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def load_data():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    feature_names = cfg["features"]
    print(f"{len(feature_names)} input features")

    extra = ["class_value", "X_mass", "weight_Central", "DeepHME_mass"]
    with uproot.open(VALIDATION_FILE) as f:
        tree = f["Events"]
        arr = tree.arrays(feature_names + extra)

    features = np.array(
        [np.asarray(arr[n]) for n in feature_names], dtype="float32"
    ).transpose()
    data = {k: np.asarray(arr[k]) for k in extra}
    return features, data


def run_model(features):
    sess = ort.InferenceSession(MODEL_PATH)
    scores = np.empty(len(features), dtype="float32")
    chunk = 200_000
    for i in range(0, len(features), chunk):
        scores[i : i + chunk] = sess.run(
            None, {"x": features[i : i + chunk]}
        )[0][:, 0]
    return scores


def set_model(model_dir):
    """Point load_data/run_model at a given trained model dir (updates the
    module globals they read). Returns the model's disco_lambda_factor."""
    global MODEL_DIR, MODEL_PATH, CONFIG_PATH
    MODEL_DIR = model_dir
    MODEL_PATH = os.path.join(model_dir, "val_combined_loss.onnx")
    CONFIG_PATH = os.path.join(model_dir, "dnn_config.yaml")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("disco_lambda_factor", "NA")


def discover_models(scan_dir, pattern):
    """Find trained model dirs under scan_dir. The ONNX/config live in the
    double-nested <name>/<name>/ dir; keep only those with a trained ONNX."""
    found = []
    for outer in sorted(glob.glob(os.path.join(scan_dir, pattern))):
        inner = os.path.join(outer, os.path.basename(outer))
        if os.path.exists(os.path.join(inner, "val_combined_loss.onnx")):
            found.append(inner)
    return found


def model_label(model_dir):
    """Short label '<production>_<TrainingN>' from a model dir path, e.g.
    prod_TTonly_Training1, so sibling productions do not overwrite each other."""
    parts = model_dir.rstrip(os.sep).split(os.sep)
    prod = next(
        (parts[i - 1] for i, p in enumerate(parts) if p.startswith("Run") and i > 0),
        "",
    )
    m = re.search(r"Training\d+", parts[-1])
    train = m.group(0) if m else parts[-1]
    return f"{prod}_{train}" if prod else train


def weighted_corr(x, y, w):
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    return cov / (sx * sy)


def step_hist(ax, vals, weights, bins, label, color, lw=1.6):
    h, edges = np.histogram(vals, bins=bins, weights=weights)
    integral = h.sum() * np.diff(edges)
    norm = h / (h.sum() * np.diff(edges)) if h.sum() > 0 else h
    ax.stairs(norm, edges, label=label, color=color, linewidth=lw)
    return norm


def make_validation_plots(output_dir, data, score, lam_label):
    os.makedirs(output_dir, exist_ok=True)

    cv = data["class_value"]
    xm = data["X_mass"]
    w = data["weight_Central"].astype("float64")
    hme = data["DeepHME_mass"].astype("float64")
    hme_ok = np.isfinite(hme) & (hme > 0)

    masks = {
        "Signal": (cv == 0) & (xm == SIGNAL_MASS),
        "TT": cv == 1,
        "DY": cv == 2,
        "Other": cv == 3,
        "All bkg": cv > 0,
    }

    # HME on-mass window from the signal peak (central 68% of weighted signal HME)
    sig_hme = hme[masks["Signal"] & hme_ok]
    sig_w = w[masks["Signal"] & hme_ok]
    order = np.argsort(sig_hme)
    cdf = np.cumsum(sig_w[order]) / sig_w.sum()
    lo = sig_hme[order][np.searchsorted(cdf, 0.16)]
    hi = sig_hme[order][np.searchsorted(cdf, 0.84)]
    lo, hi = round(lo / 10) * 10, round(hi / 10) * 10
    print(f"Signal m{SIGNAL_MASS} HME window (central 68%, rounded): [{lo}, {hi}] GeV")

    summary = [f"Model: {MODEL_PATH}", f"Validation file: {VALIDATION_FILE}", ""]
    summary.append(f"HME on-mass window (signal central 68%): [{lo}, {hi}] GeV")
    for name, m in masks.items():
        mm = m & hme_ok
        r = weighted_corr(score[mm], hme[mm], w[mm])
        summary.append(
            f"{name}: n={m.sum()}, sum(w)={w[m].sum():.1f}, "
            f"weighted corr(score, HME)={r:+.4f}"
        )
    print("\n".join(summary))
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")

    score_bins = np.linspace(0, 1, 41)
    hme_bins = np.linspace(0, 1500, 61)

    # 1. DNN score shapes per class
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name in ["Signal", "TT", "DY", "Other"]:
        m = masks[name]
        label = f"Signal m{SIGNAL_MASS}" if name == "Signal" else name
        step_hist(ax, score[m], w[m], score_bins, label, COLORS[name])
    ax.set_xlabel("DNN score")
    ax.set_ylabel("Normalized events")
    ax.legend(frameon=False)
    fig.suptitle(f"DNN score shapes, m{SIGNAL_MASS} DisCo model (lambda={lam_label})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dnn_score_shapes.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 2. HME shape in DNN score slices per background
    score_slices = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0001)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes.flat, ["TT", "DY", "Other", "All bkg"]):
        for (slo, shi), color in zip(score_slices, SCORE_SLICE_COLORS):
            m = masks[name] & hme_ok & (score >= slo) & (score < shi)
            if w[m].sum() <= 0:
                continue
            step_hist(ax, hme[m], w[m], hme_bins, f"score {slo}-{min(shi, 1.0):g}", color)
        ax.axvspan(lo, hi, color="#e34948", alpha=0.06)
        ax.set_title(name)
        ax.set_ylabel("Normalized events")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("HME mass [GeV]")
    fig.suptitle("Background HME shape in DNN score slices (should not sculpt)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "hme_shape_in_score_slices.pdf"), bbox_inches="tight"
    )
    plt.close(fig)

    # 3. Signal vs background score shape inside the HME window
    in_window = hme_ok & (hme >= lo) & (hme < hi)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name in ["Signal", "TT", "DY", "Other", "All bkg"]:
        m = masks[name] & in_window
        label = f"Signal m{SIGNAL_MASS}" if name == "Signal" else name
        lw = 2.0 if name in ("Signal", "All bkg") else 1.2
        step_hist(ax, score[m], w[m], score_bins, label, COLORS[name], lw=lw)
    ax.set_xlabel("DNN score")
    ax.set_ylabel("Normalized events")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"DNN score, events with HME in [{lo}, {hi}] GeV")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "score_in_hme_window_sig_vs_bkg.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. Cross-section-scaled score in the HME window, and S/sqrt(B) vs cut
    sig_scale = NPARITY * BR_FACTOR
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    bkg_names = ["Other", "DY", "TT"]  # smallest at the bottom of the stack
    bkg_h = [
        np.histogram(score[masks[n] & in_window], bins=score_bins,
                     weights=NPARITY * w[masks[n] & in_window])[0]
        for n in bkg_names
    ]
    bottom = np.zeros(len(score_bins) - 1)
    for name, h in zip(bkg_names, bkg_h):
        ax1.stairs(bottom + h, score_bins, baseline=bottom, fill=True,
                   color=COLORS[name], alpha=0.7, label=name)
        bottom = bottom + h
    sig_in = masks["Signal"] & in_window
    sig_h = np.histogram(score[sig_in], bins=score_bins,
                         weights=sig_scale * w[sig_in])[0]
    ax1.stairs(sig_h, score_bins, color=COLORS["Signal"], linewidth=2.0,
               label=f"Signal m{SIGNAL_MASS}")
    ax1.set_yscale("log")
    ax1.set_ylabel("Events (x-sec scaled)")
    ax1.legend(frameon=False, fontsize=8)

    # S/sqrt(B) for score >= cut, B = all background and B = TT only
    cuts = np.unique(np.concatenate([np.linspace(0.0, 0.9, 46),
                                     np.linspace(0.9, 0.999, 50)]))
    bkg_in = masks["All bkg"] & in_window
    tt_in = masks["TT"] & in_window
    rows = []
    for c in cuts:
        hi_s = score >= c
        s = sig_scale * w[sig_in & hi_s].sum()
        b = NPARITY * w[bkg_in & hi_s].sum()
        b_tt = NPARITY * w[tt_in & hi_s].sum()
        n_b = int((bkg_in & hi_s).sum())
        rows.append((c, s, b, b_tt, n_b))
    cut_arr = np.array([r[0] for r in rows])
    s_arr = np.array([r[1] for r in rows])
    b_arr = np.array([r[2] for r in rows])
    btt_arr = np.array([r[3] for r in rows])
    nb_arr = np.array([r[4] for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        z_all = np.where(b_arr > 0, s_arr / np.sqrt(b_arr), np.nan)
        z_tt = np.where(btt_arr > 0, s_arr / np.sqrt(btt_arr), np.nan)
    stable = nb_arr >= MIN_RAW_EVENTS
    z_all[~stable] = np.nan
    z_tt[~stable] = np.nan

    ax2.plot(cut_arr, z_all, color=COLORS["All bkg"], label="B = all bkg")
    ax2.plot(cut_arr, z_tt, color=COLORS["TT"], linestyle="--", label="B = TT only")
    best_line = "S/sqrt(B): no cut with enough background MC"
    if np.any(np.isfinite(z_all)):
        i = int(np.nanargmax(z_all))
        ax2.axvline(cut_arr[i], color=COLORS["Other"], linestyle=":",
                    label=f"best cut {cut_arr[i]:.3f}")
        best_line = (
            f"Best S/sqrt(B) cut (B = all bkg, HME in [{lo}, {hi}], "
            f">= {MIN_RAW_EVENTS} raw bkg events): score >= {cut_arr[i]:.4f}, "
            f"S={s_arr[i]:.4f}, B={b_arr[i]:.2f} (TT {btt_arr[i]:.2f}), "
            f"S/sqrt(B)={z_all[i]:.4f}"
        )
    ax2.set_xlabel("DNN score cut")
    ax2.set_ylabel(r"S/$\sqrt{B}$ (score $\geq$ cut)")
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"x-sec scaled score, HME in [{lo}, {hi}] GeV (x{NPARITY} parity, "
        f"lambda={lam_label})"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dnn_score_xsec.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(best_line)
    with open(os.path.join(output_dir, "summary.txt"), "a") as f:
        f.write(best_line + "\n")

    print(f"Plots written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", default=None,
        help="Single trained model dir (default: the module MODEL_DIR).",
    )
    parser.add_argument(
        "--scan-dir", default=None,
        help="Parent dir to scan for all matching models (loop mode).",
    )
    parser.add_argument(
        "--glob", default="DNN_DoubleLepton_Resolved_Disco_Training*_par0_m600",
        help="Glob (under --scan-dir) selecting model dirs to plot.",
    )
    args = parser.parse_args()

    if args.scan_dir:
        model_dirs = discover_models(args.scan_dir, args.glob)
        if not model_dirs:
            raise SystemExit(
                f"No models with val_combined_loss.onnx under "
                f"{args.scan_dir}/{args.glob}"
            )
    else:
        model_dirs = [args.model_dir or MODEL_DIR]

    print(f"{len(model_dirs)} model(s) to plot")
    for md in model_dirs:
        lam = set_model(md)
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"validation_plots_m{SIGNAL_MASS}_{model_label(md)}_lambda{lam}",
        )
        print(f"\n=== lambda={lam}  {md} -> {os.path.basename(output_dir)} ===")
        features, data = load_data()
        print("Running inference")
        score = run_model(features)
        make_validation_plots(output_dir, data, score, lam)


if __name__ == "__main__":
    main()
