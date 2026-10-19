"""Hard-coded ABCD regions as a comparable metric for DisCo trainings.

The point of this script is that every model is measured in exactly the same
regions, so two trainings can be ranked against each other. That is what
abcd_estimate.py does not do: it re-derives the HME window at runtime and picks
its working point by max Z, which lands at a different score cut for every model.

Regions come from Analysis/Linear_HME_DNN.py::LinHMEDNN.hme_bins_dict["DL"], so
the study and the analysis are talking about the same regions. For m600 the edges
are [-10, 500, 550, 650, 700, 10000], i.e. a [550, 650] core window with a buffer
band either side.

  A: score >= fc  and  HME in window        (signal region)
  B: score >= fc  and  HME in sideband
  C: score <  fc  and  HME in window
  D: score <  fc  and  HME in sideband
  N_A_pred = N_B * N_C / N_D    (background only)

Two things are deliberately not hard-coded:

  * The score cut. A raw threshold is not comparable across models - fc = 0.9
    selects a different background efficiency for every training, so closure
    differences would just be working-point differences. The cut is instead set
    per model at a fixed *weighted background efficiency*.
  * The sideband mode. "bounded" uses the buffer bands; "all" uses everything
    outside the window (the abcd_estimate.py convention). Running both measures
    how non-local the decorrelation is - see the module notes at the bottom.

This is a study-level metric for ranking trainings, not an analysis result. The
analysis figure of merit is the expected limit from combine. Treat Z here as a
relative ranking.

Usage:
    python3 abcd_metric.py --model-dir <trained model dir>
    python3 abcd_metric.py --scan-dir /eos/.../DNNTraining/prod_dcorr2/Run3_2022
"""

import os
import csv
import argparse

import numpy as np
import yaml

import plot_dnn_hme_validation as pdv
from abcd_estimate import asimov_z, dcorr2

# HME regions, from Analysis/Linear_HME_DNN.py::hme_bins_dict["DL"].
# window = the core bin; sideband = the buffer bands either side of it.
REGIONS = {
    300: {"window": (275, 325), "sideband": ((250, 275), (325, 400))},
    400: {"window": (350, 450), "sideband": ((300, 350), (450, 500))},
    500: {"window": (450, 550), "sideband": ((400, 450), (550, 600))},
    550: {"window": (475, 625), "sideband": ((400, 475), (625, 700))},
    600: {"window": (550, 650), "sideband": ((500, 550), (650, 700))},
    650: {"window": (550, 700), "sideband": ((500, 550), (700, 800))},
    700: {"window": (625, 750), "sideband": ((550, 625), (750, 800))},
    800: {"window": (700, 850), "sideband": ((650, 700), (850, 950))},
    900: {"window": (800, 950), "sideband": ((700, 800), (950, 1050))},
    1000: {"window": (875, 1075), "sideband": ((750, 875), (1075, 1150))},
}

# Fixed weighted background efficiencies at which every model is compared.
WORKING_POINTS = [0.10, 0.01, 0.001]

# Minimum raw MC events per region for the prediction to be trusted.
MIN_RAW = 20

# Branching factor applied to signal (matches DNN_Class_HistTuples.py).
BR_FACTOR = 0.0264215349425664

# The validation file is one parity of NPARITY. Yields are scaled up to
# approximate the full dataset; set to 1 to quote this file as-is.
NPARITY = 4

# class_value from dataset_setup_doubleLep_resolved_v2605a.yaml
PROCESSES = {"TT": 1, "DY": 2, "Other": 3}

CSV_FIELDS = [
    "model", "lambda", "sideband_mode", "eff_bkg", "fc",
    "nA", "nB", "nC", "nD", "stable",
    "B_true", "B_pred", "B_pred_err", "closure",
    "closure_TT", "closure_DY", "closure_Other",
    "S", "Z_stat", "Z_syst",
    "dcorr2_TT", "null_TT", "dcorr2_DY", "null_DY",
    "dcorr2_Other", "null_Other", "dcorr2_all", "null_all",
]


def weighted_quantile(values, q, weights):
    """Value below which fraction q of the total weight lies."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return np.interp(q * cw[-1], cw, v)


def region_masks(hme, mass, sideband_mode):
    """in_window / in_sideband for the hard-coded regions of this mass."""
    reg = REGIONS[mass]
    ok = np.isfinite(hme) & (hme > 0)
    lo, hi = reg["window"]
    in_window = ok & (hme >= lo) & (hme < hi)
    if sideband_mode == "all":
        in_sideband = ok & ~in_window
    else:
        in_sideband = np.zeros_like(ok)
        for slo, shi in reg["sideband"]:
            in_sideband |= ok & (hme >= slo) & (hme < shi)
    return ok, in_window, in_sideband


def abcd_yields(sel, score, fc, in_window, in_sideband, w):
    """Weighted yield, squared-weight sum and raw count for A/B/C/D."""
    hi = score >= fc
    regions = {
        "A": sel & hi & in_window,
        "B": sel & hi & in_sideband,
        "C": sel & ~hi & in_window,
        "D": sel & ~hi & in_sideband,
    }
    out = {}
    for name, m in regions.items():
        ww = w[m]
        out[name] = (ww.sum(), np.square(ww).sum(), int(m.sum()))
    return out


def closure_of(y):
    """(N_B * N_C / N_D) / N_A, or nan if any region is empty."""
    yA, yB, yC, yD = (y[k][0] for k in "ABCD")
    if min(yA, yB, yC, yD) <= 0:
        return np.nan
    return (yB * yC / yD) / yA


def dcorr2_with_null(score, hme, w, seed=0):
    """dCorr^2(score, HME) and the permutation null at the same sample size.

    The null is what a genuinely decorrelated sample of this size would give, so
    it is the reference the measured value has to be compared against. "Small"
    is not meaningful on its own.

    Negative-weight events are dropped. Distance correlation is defined over a
    probability measure; with negative weights the double centring is unbounded
    and the estimator destabilises. DY is 30% negative weight (amcatnloFXFX),
    which inflated its apparent dCorr^2 ~5x and its null ~35x. Dropping them
    also matches what the training optimises: add_weight_file
    (create_dataset.py:338) zeroes negative weights before they reach the DisCo
    penalty, so the model never sees them either. Yields keep the real weights -
    only this diagnostic clips.
    """
    keep = w > 0
    score, hme, w = score[keep], hme[keep], w[keep]
    if len(score) < 50:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    value = dcorr2(score, hme, w, seed=seed)
    null = dcorr2(score, rng.permutation(hme), w, seed=seed + 1)
    return value, null


def evaluate(model_dir, mass, sideband_mode, features, data, score, lam):
    """One row per working point for a single model."""
    cv = data["class_value"]
    xm = data["X_mass"]
    w = data["weight_Central"].astype("float64")
    hme = data["DeepHME_mass"].astype("float64")

    ok, in_window, in_sideband = region_masks(hme, mass, sideband_mode)
    bkg = (cv > 0) & ok
    sig = (cv == 0) & (xm == mass) & ok

    # Decorrelation diagnostics are working-point independent, so compute once.
    dcorr = {}
    for name, cval in list(PROCESSES.items()) + [("all", None)]:
        m = bkg if cval is None else (bkg & (cv == cval))
        dcorr[name] = dcorr2_with_null(score[m], hme[m], w[m])

    rows = []
    for eff in WORKING_POINTS:
        fc = weighted_quantile(score[bkg], 1.0 - eff, w[bkg])
        y = abcd_yields(bkg, score, fc, in_window, in_sideband, w)

        (yA, _, nA), (yB, w2B, nB) = y["A"], y["B"]
        (yC, w2C, nC), (yD, w2D, nD) = y["C"], y["D"]
        if min(yB, yC, yD) <= 0:
            continue

        b_pred = NPARITY * (yB * yC / yD)
        b_true = NPARITY * yA
        clos = closure_of(y)

        # Sideband stat error on B*C/D, reduced by sqrt(NPARITY) for full stats.
        relerr2 = w2B / (yB * yB) + w2C / (yC * yC) + w2D / (yD * yD)
        stat = b_pred * np.sqrt(relerr2 / NPARITY)
        # Charge the non-closure to the background uncertainty rather than only
        # reporting it, so a sculpting model cannot win on Z alone.
        nonclos = abs(1.0 - clos) * b_pred if np.isfinite(clos) else 0.0
        sigma_b = np.hypot(stat, nonclos)

        s = NPARITY * BR_FACTOR * w[sig & (score >= fc) & in_window].sum()

        per_proc = {}
        for name, cval in PROCESSES.items():
            yp = abcd_yields(bkg & (cv == cval), score, fc,
                             in_window, in_sideband, w)
            per_proc[name] = closure_of(yp)

        rows.append({
            "model": os.path.basename(model_dir.rstrip(os.sep)),
            "lambda": lam,
            "sideband_mode": sideband_mode,
            "eff_bkg": eff,
            "fc": round(float(fc), 5),
            "nA": nA, "nB": nB, "nC": nC, "nD": nD,
            "stable": bool(min(nA, nB, nC, nD) >= MIN_RAW),
            "B_true": round(b_true, 3),
            "B_pred": round(b_pred, 3),
            "B_pred_err": round(float(stat), 3),
            "closure": round(float(clos), 4),
            "closure_TT": round(float(per_proc["TT"]), 4),
            "closure_DY": round(float(per_proc["DY"]), 4),
            "closure_Other": round(float(per_proc["Other"]), 4),
            "S": round(float(s), 4),
            "Z_stat": round(float(asimov_z(s, b_pred)), 4),
            "Z_syst": round(float(asimov_z(s, b_pred, sigma_b)), 4),
            **{f"{k}_{name}": round(float(v), 5)
               for name in list(PROCESSES) + ["all"]
               for k, v in zip(("dcorr2", "null"), dcorr[name])},
        })
    return rows


_CACHE = {}


def load_for(model_dir):
    """Features/data for a model, reusing the load when the feature list matches."""
    lam = pdv.set_model(model_dir)
    with open(os.path.join(model_dir, "dnn_config.yaml")) as f:
        key = tuple(yaml.safe_load(f)["features"])
    if key not in _CACHE:
        _CACHE.clear()  # only one feature set in memory at a time
        _CACHE[key] = pdv.load_data()
    features, data = _CACHE[key]
    return features, data, lam


def print_table(rows):
    hdr = (f"{'eff_b':>7s} {'fc':>7s} {'rawA':>6s} {'rawB':>6s} {'B_pred':>9s} "
           f"{'closure':>8s} {'TT':>7s} {'DY':>7s} {'Other':>7s} "
           f"{'S':>6s} {'Z_syst':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "" if r["stable"] else "  <- LOW STATS"
        print(f"{r['eff_bkg']:7.3%} {r['fc']:7.4f} {r['nA']:6d} {r['nB']:6d} "
              f"{r['B_pred']:9.2f} {r['closure']:8.3f} {r['closure_TT']:7.3f} "
              f"{r['closure_DY']:7.3f} {r['closure_Other']:7.3f} "
              f"{r['S']:6.3f} {r['Z_syst']:7.4f}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=None, help="A single trained model dir.")
    p.add_argument("--scan-dir", default=None,
                   help="Parent dir; every matching model under it is evaluated.")
    p.add_argument("--glob", default="DNN_DoubleLepton_Resolved_Disco_Training*_par0_m600",
                   help="Glob under --scan-dir selecting model dirs.")
    p.add_argument("--mass", type=int, default=600, help="Signal mass point.")
    p.add_argument("--sideband", default="bounded", choices=["bounded", "all", "both"],
                   help="'bounded' = buffer bands, 'all' = everything outside the "
                        "window, 'both' = run each and report the gap.")
    p.add_argument("--out", default="abcd_metric.csv", help="Output CSV path.")
    args = p.parse_args()

    if args.mass not in REGIONS:
        raise SystemExit(f"No hard-coded regions for m{args.mass}; add it to REGIONS.")

    if args.scan_dir:
        model_dirs = pdv.discover_models(args.scan_dir, args.glob)
        if not model_dirs:
            raise SystemExit(f"No models under {args.scan_dir}/{args.glob}")
    elif args.model_dir:
        model_dirs = [args.model_dir]
    else:
        raise SystemExit("Give --model-dir or --scan-dir")

    modes = ["bounded", "all"] if args.sideband == "both" else [args.sideband]
    reg = REGIONS[args.mass]
    print(f"m{args.mass} window {reg['window']}, sideband bands {reg['sideband']}")
    print(f"{len(model_dirs)} model(s)\n")

    all_rows = []
    for md in model_dirs:
        features, data, lam = load_for(md)
        score = pdv.run_model(features)
        for mode in modes:
            rows = evaluate(md, args.mass, mode, features, data, score, lam)
            all_rows += rows
            print(f"=== {os.path.basename(md)}  lambda={lam}  sideband={mode} ===")
            print_table(rows)
            print()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {args.out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
