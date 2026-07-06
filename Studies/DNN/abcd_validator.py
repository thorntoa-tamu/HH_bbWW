"""Post-training validation of the ABCD(isCo) method assumptions.

The DNN is trained with DisCo regularization (Single DisCo: the classifier
output f is decorrelated against mbb on background, see DiscoModel in
DNN_Class.py), but nothing in the pipeline verifies the conditions that make
the ABCD background estimate N_A = N_B * N_C / N_D unbiased. Following
Kasieczka, Nachman, Schwartz, Shih, "ABCDisCo: Automating the ABCD Method
with Machine Learning" (arXiv:2007.14400), this script checks, per parametric
mass point:

  1. Independence of f = DNN signal score and g = mbb on background:
       - ABCD closure lambda = (N_B,b * N_C,b / N_D,b) / N_A,b  (Eq. 2.2/2.3)
         scanned over DNN thresholds, with MC-stat errors.
       - weighted distance correlation dCorr^2(f, mbb) (Appendix A), with a
         permutation baseline to calibrate the finite-sample positive bias.
  2. Signal contamination in the control regions (Eq. 2.7):
       delta_i = N_i,s / N_i,b  << 1  for i = B, C, D.
  3. Normalized signal contamination (Eq. 2.8/2.9):
       r = (delta_B + delta_C - delta_D) / delta_A,  |r| << 1.
     Note r is invariant under a global rescaling of the signal cross
     section, so it is meaningful even when the signal normalization in
     weight_MC_Lumi_pu is an arbitrary reference value; the delta_i scale
     linearly with the assumed cross section (use --signal-scale to test).

It also produces the performance metrics shown in the paper, computed on the
analysis dataset:
       - signal efficiency vs background rejection (ROC, Fig. 5/9 style),
         both for a cut on f alone and for the full region-A selection;
       - ABCD closure vs background rejection (Fig. 8 left);
       - normalized signal contamination r vs background rejection
         (Fig. 7 / Fig. 8 right / Fig. 12 style).

With --model-dir instead of --model-name, every epoch checkpoint written by
ModelCheckpoint (epoch_N.onnx, plus best.onnx / stage1.onnx) is validated and
an across-epoch scan (epoch_scan_m{mass}.csv/pdf, the paper's Fig. 7 workflow
of choosing the training epoch by ABCD criteria) is produced on top of the
per-model outputs. This is the mode used by ABCDValidationTask (law task in
Studies/DNN/tasks.py) via the ABCD_Validator_Condor.py wrapper.

The ABCD plane mirrors the pipeline conventions (weight_calculator.py):
f is cut at a scanned threshold fc, and g = mbb is split into the window
[mbb_low, mbb_high) (signal side) and the high sideband [mbb_high, mbb_max)
(the low sideband [mbb_min, mbb_low) is dropped by default, as in training).

  A = f >= fc and mbb in window     (signal region)
  B = f >= fc and mbb in sideband
  C = f <  fc and mbb in window
  D = f <  fc and mbb in sideband

Inputs mirror DNN_Validator_Condor.py: a batchfile (Events tree), the
matching weight file (weight_tree, gives class_target: 0 = signal), the
trained ONNX model(s) and the model config yaml (either the training setup
yaml with grouped listfeatures or the dnn_config.yaml written by train_dnn
with flattened listfeatures). Physics yields use the MC event weight branch
(weight_base or weight_MC_Lumi_pu, auto-detected), NOT the training
class/adv weights: class_weight rebalances signal to background and
adv_weight zeroes the signal, so neither can measure delta_i or r.

Example (single model):
  python3 Studies/DNN/abcd_validator.py \
      --validation-file batchfile2.root \
      --validation-weight-file weightfile2.root \
      --model-name best.onnx --model-config dnn_config.yaml \
      --hme-friend-file batchfile2_HME_Friend.root \
      --output-folder abcd_validation

Example (epoch scan over a training output folder):
  python3 Studies/DNN/abcd_validator.py ... \
      --model-dir <training_output_folder> --epoch-step 2
"""

import argparse
import csv
import os
import re

import awkward as ak
import numpy as np
import uproot
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import onnxruntime as ort

PHYSICS_WEIGHT_CANDIDATES = ["weight_base", "weight_MC_Lumi_pu"]
DEFAULT_MBB_BRANCH = "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino"
EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)\.onnx$")


# =========================================================
# CONFIG AND FEATURE LOADING
# =========================================================
def flatten_listfeatures(listfeatures):
    """Normalize listfeatures to [(branch_name, index), ...] preserving order.

    Accepts both formats found in the pipeline:
      - training setup yaml: [[[name1, name2, ...], index], ...]
      - dnn_config.yaml (written by train_dnn): [[name, index], ...]
    """
    flat = []
    if listfeatures is None:
        return flat
    for entry in listfeatures:
        head, index = entry
        if isinstance(head, str):
            flat.append((head, int(index)))
        else:
            flat.extend((name, int(index)) for name in head)
    return flat


def get_config_value(config, *keys, default=None):
    """Return the first of *keys present in config (checks model_setup too)."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    model_setup = config.get("model_setup", {}) or {}
    for key in keys:
        if key in model_setup and model_setup[key] is not None:
            return model_setup[key]
    return default


def load_features(validation_file, config, hme_friend_file=None):
    """Build the DNN input matrix with the same column order as DataWrapper.

    Order: features, listfeatures (padded per index), highlevelfeatures,
    hmefeatures (from the friend tree), then one slot for the parametric
    mass which the caller fills per mass point. Returns (X, mbb, X_mass)
    with X float64 of shape (nEvents, nFeatures + 1).
    """
    feature_names = list(get_config_value(config, "features") or [])
    listfeature_pairs = flatten_listfeatures(get_config_value(config, "listfeatures"))
    hlv_names = list(get_config_value(config, "highlevelfeatures") or [])
    hme_names = list(get_config_value(config, "hmefeatures") or [])
    mbb_branch = get_config_value(config, "mbb_branch", default=DEFAULT_MBB_BRANCH)

    branches_to_load = list(
        dict.fromkeys(
            feature_names + [name for name, _ in listfeature_pairs] + hlv_names
        )
    )
    branches_to_load += [mbb_branch, "X_mass"]

    print(f"Reading {validation_file}")
    with uproot.open(validation_file) as file:
        tree = file["Events"]
        branches = tree.arrays(branches_to_load)

    columns = []
    for name in feature_names:
        columns.append(np.asarray(branches[name], dtype=np.float64))
    for name, index in listfeature_pairs:
        padded = ak.fill_none(ak.pad_none(branches[name], index + 1, axis=1), 0.0)
        columns.append(np.asarray(padded[:, index], dtype=np.float64))
    for name in hlv_names:
        columns.append(np.asarray(branches[name], dtype=np.float64))

    if hme_names:
        if hme_friend_file is None:
            raise ValueError(
                "Model config requests hmefeatures but no --hme-friend-file given"
            )
        print(f"Reading HME friend {hme_friend_file}")
        with uproot.open(hme_friend_file) as file:
            hme_branches = file["Events"].arrays(hme_names)
        for name in hme_names:
            columns.append(np.asarray(hme_branches[name], dtype=np.float64))

    mbb = np.asarray(branches[mbb_branch], dtype=np.float64)
    x_mass = np.asarray(branches["X_mass"])

    n_events = len(mbb)
    use_parametric = bool(
        get_config_value(config, "use_parametric", "UseParametric", default=False)
    )
    n_columns = len(columns) + (1 if use_parametric else 0)
    X = np.empty((n_events, n_columns), dtype=np.float64)
    for i, column in enumerate(columns):
        X[:, i] = column

    print(f"Feature matrix: {X.shape} (parametric slot included: {use_parametric})")
    return X, mbb, x_mass, use_parametric


def load_weight_file(weight_file):
    with uproot.open(weight_file) as file:
        tree = file["weight_tree"]
        arrays = tree.arrays(["class_target"])
    return np.asarray(arrays["class_target"], dtype=np.int64)


def load_physics_weight(validation_file, branch=None):
    with uproot.open(validation_file) as file:
        tree = file["Events"]
        keys = set(tree.keys())
        if branch is None:
            for candidate in PHYSICS_WEIGHT_CANDIDATES:
                if candidate in keys:
                    branch = candidate
                    break
        if branch is None or branch not in keys:
            raise ValueError(
                f"No physics weight branch found (tried {PHYSICS_WEIGHT_CANDIDATES})."
                " Pass --physics-weight-branch."
            )
        weights = np.asarray(tree[branch].array(), dtype=np.float64)
    print(f"Physics weights from '{branch}': sum={weights.sum():.6g}")
    return weights, branch


# =========================================================
# MODELS AND INFERENCE
# =========================================================
def discover_models(model_dir, epoch_step=1):
    """Find ONNX models in a training output folder.

    Returns [(label, path), ...]: epoch checkpoints epoch_N.onnx in epoch
    order (thinned to every epoch_step-th, the last one always kept),
    followed by the final models (best.onnx from the DisCo trainer,
    stage1.onnx from the HistTuples trainer, epoch_final.onnx fallback).
    Duplicates via symlinks (best.onnx -> epoch_N.onnx) are dropped.
    """
    files = sorted(os.listdir(model_dir))
    epochs = sorted(
        (int(match.group(1)), name)
        for name in files
        if (match := EPOCH_FILE_RE.match(name))
    )
    kept = epochs[:: max(1, epoch_step)]
    if epochs and epochs[-1] not in kept:
        kept.append(epochs[-1])

    models = [(f"epoch_{n}", os.path.join(model_dir, name)) for n, name in kept]
    for file_name, label in (
        ("best.onnx", "best"),
        ("stage1.onnx", "stage1"),
        ("epoch_final.onnx", "final"),
    ):
        path = os.path.join(model_dir, file_name)
        if os.path.exists(path):
            models.append((label, path))

    seen = set()
    unique_models = []
    for label, path in models:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        unique_models.append((label, path))

    if not unique_models:
        unique_models = [
            (os.path.splitext(name)[0], os.path.join(model_dir, name))
            for name in files
            if name.endswith(".onnx")
        ]
    if not unique_models:
        raise ValueError(f"No ONNX models found in {model_dir}")
    return unique_models


def epoch_number(label):
    match = re.match(r"^epoch_(\d+)$", label)
    return int(match.group(1)) if match else None


def primary_label(models):
    """Model used for the full plots and the dCorr check: the final one."""
    labels = [label for label, _ in models]
    for preferred in ("best", "stage1", "final"):
        if preferred in labels:
            return preferred
    return labels[-1]


def make_session(model_name):
    """ONNX session preferring the CUDA provider when the env provides it."""
    available = ort.get_available_providers()
    providers = [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ] or None
    sess = ort.InferenceSession(model_name, providers=providers)
    print(f"Loaded {model_name} (providers: {sess.get_providers()})")
    return sess


def run_inference(sess, X, score_class=0, chunk_size=200_000):
    input_name = sess.get_inputs()[0].name
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        pred = sess.run(None, {input_name: X[start:stop]})[0]
        scores[start:stop] = pred[:, score_class]
    return scores


# =========================================================
# STATISTICS
# =========================================================
def weighted_yield(weights, mask):
    """Yield and its MC-stat variance (sum w, sum w^2) under mask."""
    w = weights[mask]
    return w.sum(), (w * w).sum()


def safe_ratio(num, den):
    return num / den if den != 0 else float("nan")


def region_table(score, fc, in_window, in_sideband, is_signal, weights):
    """Yields per ABCD region for signal and background.

    Returns {region: {"s": (N, var), "b": (N, var)}} for regions A, B, C, D.
    """
    pass_cut = score >= fc
    masks = {
        "A": pass_cut & in_window,
        "B": pass_cut & in_sideband,
        "C": ~pass_cut & in_window,
        "D": ~pass_cut & in_sideband,
    }
    table = {}
    for region, mask in masks.items():
        table[region] = {
            "s": weighted_yield(weights, mask & is_signal),
            "b": weighted_yield(weights, mask & ~is_signal),
        }
    return table


def abcd_conditions(table, signal_scale=1.0):
    """Compute the ABCDisCo validity conditions from region yields.

    Returns a dict with:
      delta_{A,B,C,D} and errors      (Eq. 2.7, scaled by signal_scale)
      r and error                     (Eq. 2.8; invariant under signal_scale)
      closure lambda = N_A,b^pred / N_A,b and error, non-closure Z
      N_A_pred_b, N_A_pred_all, signal_recovery = (N_A,a - N_A^pred,a)/N_A,s
    Regions are disjoint, so their yields are treated as independent.
    """
    N = {region: {} for region in "ABCD"}
    rel_var = {region: {} for region in "ABCD"}
    for region in "ABCD":
        for kind in ("s", "b"):
            value, variance = table[region][kind]
            if kind == "s":
                value, variance = value * signal_scale, variance * signal_scale**2
            N[region][kind] = value
            rel_var[region][kind] = variance / value**2 if value != 0 else float("nan")

    out = {}
    for region in "ABCD":
        delta = safe_ratio(N[region]["s"], N[region]["b"])
        out[f"delta_{region}"] = delta
        out[f"delta_{region}_err"] = (
            abs(delta) * np.sqrt(rel_var[region]["s"] + rel_var[region]["b"])
            if np.isfinite(delta)
            else float("nan")
        )

    # r = (delta_B + delta_C - delta_D) / delta_A  (Eq. 2.8)
    dA, dB, dC, dD = (out[f"delta_{k}"] for k in "ABCD")
    if np.isfinite([dA, dB, dC, dD]).all() and dA != 0:
        r = (dB + dC - dD) / dA
        var_numerator = sum(out[f"delta_{k}_err"] ** 2 for k in "BCD")
        r_err = np.sqrt(var_numerator / dA**2 + r**2 * (out["delta_A_err"] / dA) ** 2)
    else:
        r, r_err = float("nan"), float("nan")
    out["r"] = r
    out["r_err"] = r_err

    # Background-only closure (Eq. 2.3): lambda = (N_B,b N_C,b / N_D,b) / N_A,b
    NA_b, NB_b, NC_b, ND_b = (N[k]["b"] for k in "ABCD")
    if min(NA_b, NB_b, NC_b, ND_b) > 0:
        NA_pred_b = NB_b * NC_b / ND_b
        closure = NA_pred_b / NA_b
        rel_var_pred = rel_var["B"]["b"] + rel_var["C"]["b"] + rel_var["D"]["b"]
        closure_err = closure * np.sqrt(rel_var_pred + rel_var["A"]["b"])
        sigma_delta = np.sqrt(NA_pred_b**2 * rel_var_pred + NA_b**2 * rel_var["A"]["b"])
        z_nonclosure = (
            (NA_pred_b - NA_b) / sigma_delta if sigma_delta > 0 else float("nan")
        )
    else:
        NA_pred_b, closure, closure_err, z_nonclosure = (float("nan"),) * 4
    out["N_A_pred_b"] = NA_pred_b
    out["closure"] = closure
    out["closure_err"] = closure_err
    out["z_nonclosure"] = z_nonclosure

    # Prediction as it would run on data (Eq. 2.2, all = s + b): the part of
    # the observable excess that survives is signal_recovery ~ 1 - r + non-closure
    NA_all = N["A"]["s"] + N["A"]["b"]
    NB_all = N["B"]["s"] + N["B"]["b"]
    NC_all = N["C"]["s"] + N["C"]["b"]
    ND_all = N["D"]["s"] + N["D"]["b"]
    if min(NB_all, NC_all, ND_all) > 0 and N["A"]["s"] > 0:
        NA_pred_all = NB_all * NC_all / ND_all
        signal_recovery = (NA_all - NA_pred_all) / N["A"]["s"]
    else:
        NA_pred_all, signal_recovery = float("nan"), float("nan")
    out["N_A_pred_all"] = NA_pred_all
    out["signal_recovery"] = signal_recovery

    for region in "ABCD":
        out[f"N_{region}_s"] = N[region]["s"]
        out[f"N_{region}_b"] = N[region]["b"]
    return out


def distance_corr_numpy(var_1, var_2, weights, power=2):
    """Weighted distance correlation, numpy port of DNN_Class.distance_corr.

    weights are normalized internally so that sum(w) = N, matching the
    'normedweight' convention of the training loss (Appendix A of the paper).
    """
    var_1 = np.asarray(var_1, dtype=np.float64)
    var_2 = np.asarray(var_2, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(var_1)
    w = w * (n / w.sum())

    amat = np.abs(var_1[:, None] - var_1[None, :])
    bmat = np.abs(var_2[:, None] - var_2[None, :])

    amatavg = (amat * w[None, :]).mean(axis=1)
    bmatavg = (bmat * w[None, :]).mean(axis=1)

    Amat = amat - amatavg[None, :] - amatavg[:, None] + (amatavg * w).mean()
    Bmat = bmat - bmatavg[None, :] - bmatavg[:, None] + (bmatavg * w).mean()

    ABavg = np.abs((Amat * Bmat * w[None, :]).mean(axis=1))
    AAavg = (Amat * Amat * w[None, :]).mean(axis=1)
    BBavg = (Bmat * Bmat * w[None, :]).mean(axis=1)

    dcov2 = (ABavg * w).mean()
    denom = abs((AAavg * w).mean() * (BBavg * w).mean()) + 1e-12
    if power == 1:
        return dcov2 / np.sqrt(denom)
    if power == 2:
        return dcov2**2 / denom
    return (dcov2 / np.sqrt(denom)) ** power


def disco_check(score, mbb, weights, mask, n_sub=4000, n_boot=8, seed=1234):
    """dCorr^2(f, mbb) on background with a shuffled-mbb permutation baseline.

    The weighted dCorr estimator has a positive finite-sample bias, so the
    observed value is compared against the same estimator on (f, permuted
    mbb) subsamples, which is the independent-by-construction reference.
    Returns dict with mean/std of both.
    """
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(mask & (weights != 0))
    values, baselines = [], []
    for _ in range(n_boot):
        sub = rng.choice(indices, size=min(n_sub, len(indices)), replace=False)
        f_sub, m_sub, w_sub = score[sub], mbb[sub], weights[sub]
        values.append(distance_corr_numpy(f_sub, m_sub, w_sub, power=2))
        baselines.append(
            distance_corr_numpy(f_sub, rng.permutation(m_sub), w_sub, power=2)
        )
    return {
        "dcorr2": float(np.mean(values)),
        "dcorr2_std": float(np.std(values)),
        "dcorr2_perm": float(np.mean(baselines)),
        "dcorr2_perm_std": float(np.std(baselines)),
        "n_sub": int(min(n_sub, len(indices))),
        "n_boot": n_boot,
    }


# =========================================================
# THRESHOLD SCAN
# =========================================================
def scan_thresholds(score, in_window, in_sideband, is_signal, weights, args):
    """Scan DNN cuts; compute efficiencies, rejections and ABCD conditions.

    Efficiencies are defined within the ABCD acceptance (window + sideband):
      eff_sig_A = N_A,s / N_s(acceptance)   [both cuts]
      eff_bkg_A = N_A,b / N_b(acceptance),  rejection_A = 1 / eff_bkg_A
      eff_sig_f, eff_bkg_f: cut on f alone within the acceptance (ROC).
    """
    acceptance = in_window | in_sideband
    total_s = weights[acceptance & is_signal].sum() * args.signal_scale
    total_b = weights[acceptance & ~is_signal].sum()

    signal_scores = score[acceptance & is_signal]
    quantiles = np.linspace(0.005, 0.995, args.n_scan)
    cuts = np.unique(np.quantile(signal_scores, quantiles))

    rows = []
    for fc in cuts:
        table = region_table(score, fc, in_window, in_sideband, is_signal, weights)
        row = {"fc": float(fc)}
        row.update(abcd_conditions(table, signal_scale=args.signal_scale))

        pass_cut = (score >= fc) & acceptance
        pass_s = weights[pass_cut & is_signal].sum() * args.signal_scale
        pass_b = weights[pass_cut & ~is_signal].sum()
        row["eff_sig_f"] = safe_ratio(pass_s, total_s)
        row["eff_bkg_f"] = safe_ratio(pass_b, total_b)
        row["eff_sig_A"] = safe_ratio(row["N_A_s"], total_s)
        row["eff_bkg_A"] = safe_ratio(row["N_A_b"], total_b)
        row["rejection_A"] = safe_ratio(1.0, row["eff_bkg_A"])
        row["rejection_f"] = safe_ratio(1.0, row["eff_bkg_f"])
        rows.append(row)
    return rows


def pick_working_points(rows, eff_targets):
    """Nearest scan point to each target signal efficiency (region A)."""
    points = {}
    effs = np.array([row["eff_sig_A"] for row in rows])
    for target in eff_targets:
        if not np.isfinite(effs).any():
            continue
        index = int(np.nanargmin(np.abs(effs - target)))
        points[target] = rows[index]
    return points


def evaluate_flags(wp_row, disco_result, args):
    """PASS/WARN flags for the paper's conditions at one working point."""
    max_delta_ctrl = np.nanmax(
        [wp_row["delta_B"], wp_row["delta_C"], wp_row["delta_D"]]
    )
    flags = {
        "closure_ok": bool(abs(wp_row["closure"] - 1.0) < args.closure_tol),
        "delta_ok": bool(max_delta_ctrl < args.delta_warn),
        "r_ok": bool(abs(wp_row["r"]) < args.r_warn),
        "max_delta_ctrl": float(max_delta_ctrl),
    }
    if disco_result is not None:
        flags["dcorr_ok"] = bool(
            disco_result["dcorr2"]
            < max(
                args.dcorr_abs_ok,
                args.dcorr_warn_ratio * disco_result["dcorr2_perm"],
            )
        )
    return flags


# =========================================================
# OUTPUTS
# =========================================================
CSV_COLUMNS = [
    "fc",
    "eff_sig_A",
    "eff_bkg_A",
    "rejection_A",
    "eff_sig_f",
    "eff_bkg_f",
    "rejection_f",
    "closure",
    "closure_err",
    "z_nonclosure",
    "delta_A",
    "delta_A_err",
    "delta_B",
    "delta_B_err",
    "delta_C",
    "delta_C_err",
    "delta_D",
    "delta_D_err",
    "r",
    "r_err",
    "signal_recovery",
    "N_A_s",
    "N_A_b",
    "N_B_s",
    "N_B_b",
    "N_C_s",
    "N_C_b",
    "N_D_s",
    "N_D_b",
    "N_A_pred_b",
    "N_A_pred_all",
]

EPOCH_CSV_COLUMNS = [
    "label",
    "epoch",
    "target_eff",
    "fc",
    "eff_sig_A",
    "rejection_A",
    "closure",
    "closure_err",
    "z_nonclosure",
    "r",
    "r_err",
    "delta_B",
    "delta_C",
    "delta_D",
    "signal_recovery",
]


def write_csv(rows, path, columns):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_text_page(pdf, label, mass, working_points, disco_result, flags, args):
    lines = [
        f"ABCD validation summary  --  model '{label}', mass point {mass} GeV",
        "",
        f"Regions: window [{args.mbb_low}, {args.mbb_high}) GeV,"
        f" sideband [{args.mbb_high}, {args.mbb_max}) GeV (mbb branch: {args.mbb_branch})",
        f"Signal scale applied to delta_i: {args.signal_scale} (r is scale-invariant)",
        "",
    ]
    if disco_result is not None:
        lines += [
            f"Independence:  dCorr^2(f, mbb | bkg) = {disco_result['dcorr2']:.5f}"
            f" +- {disco_result['dcorr2_std']:.5f}",
            f"  permutation baseline (independent case) = {disco_result['dcorr2_perm']:.5f}"
            f" +- {disco_result['dcorr2_perm_std']:.5f}"
            f"   [n_sub={disco_result['n_sub']}, n_boot={disco_result['n_boot']}]",
            "",
        ]
    lines += [
        "Working points (nearest scan point to target eff_sig in region A):",
        "",
        "  target  fc      eff_s   rej_A    closure(+-err)  Z_nc   "
        "delta_B   delta_C   delta_D    r(+-err)      recovery",
    ]
    for target, row in sorted(working_points.items()):
        lines.append(
            f"  {target:5.0%}  {row['fc']:.4f}  {row['eff_sig_A']:.3f} "
            f" {row['rejection_A']:8.1f}  {row['closure']:.3f}+-{row['closure_err']:.3f} "
            f" {row['z_nonclosure']:+5.1f}  {row['delta_B']:.2e}  {row['delta_C']:.2e} "
            f" {row['delta_D']:.2e}  {row['r']:+.3f}+-{row['r_err']:.3f}  "
            f"{row['signal_recovery']:+.3f}"
        )
    lines += [
        "",
        "Conditions at the primary working point "
        f"(eff_sig ~ {args.primary_eff:.0%}):",
        f"  [{'PASS' if flags['closure_ok'] else 'WARN'}] ABCD closure within "
        f"{args.closure_tol:.0%}",
        f"  [{'PASS' if flags['delta_ok'] else 'WARN'}] control-region contamination "
        f"delta_i < {args.delta_warn} (max = {flags['max_delta_ctrl']:.3g})",
        f"  [{'PASS' if flags['r_ok'] else 'WARN'}] normalized signal contamination "
        f"|r| < {args.r_warn}",
    ]
    if "dcorr_ok" in flags:
        lines.append(
            f"  [{'PASS' if flags['dcorr_ok'] else 'WARN'}] dCorr^2 consistent with "
            f"independence (< {args.dcorr_warn_ratio}x permutation baseline)"
        )
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.03, 0.97, "\n".join(lines), family="monospace", fontsize=8, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def make_plots(
    pdf, mass, rows, score, mbb, in_window, in_sideband, is_signal, weights, args
):
    acceptance = in_window | in_sideband
    sig, bkg = acceptance & is_signal, acceptance & ~is_signal

    def arr(key):
        return np.array([row[key] for row in rows], dtype=np.float64)

    # 1) mbb spectra with region boundaries
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(args.mbb_min, args.mbb_max, 60)
    ax.hist(
        mbb[bkg],
        bins=bins,
        weights=weights[bkg],
        density=True,
        histtype="step",
        label="Background",
        color="k",
    )
    ax.hist(
        mbb[sig],
        bins=bins,
        weights=weights[sig],
        density=True,
        histtype="stepfilled",
        alpha=0.4,
        label=f"Signal m{mass}",
        color="orange",
    )
    for x in (args.mbb_low, args.mbb_high):
        ax.axvline(x, color="r", linestyle="--")
    ax.set_xlabel("mbb [GeV]")
    ax.set_ylabel("normalized")
    ax.set_title(f"mbb regions, m={mass} GeV")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # 2) background score shape: window vs sideband (decorrelation eyeball check)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(
        score[bkg & in_window],
        bins=bins,
        weights=weights[bkg & in_window],
        density=True,
        histtype="step",
        label="bkg, mbb window",
        color="b",
    )
    ax.hist(
        score[bkg & in_sideband],
        bins=bins,
        weights=weights[bkg & in_sideband],
        density=True,
        histtype="step",
        label="bkg, mbb sideband",
        color="r",
    )
    ax.hist(
        score[sig & in_window],
        bins=bins,
        weights=weights[sig & in_window],
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="sig, mbb window",
        color="orange",
    )
    ax.set_xlabel("DNN signal score f")
    ax.set_ylabel("normalized")
    ax.set_yscale("log")
    ax.set_title(f"Score shapes, m={mass} GeV")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # 3) ROC: signal efficiency vs background rejection (Fig. 5/9 style)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(arr("eff_sig_f"), arr("rejection_f"), label="cut on f only")
    ax.plot(
        arr("eff_sig_A"),
        arr("rejection_A"),
        "--",
        label="ABCD region A (f cut x mbb window)",
    )
    ax.set_xlabel("signal efficiency")
    ax.set_ylabel("background rejection (1/eff_bkg)")
    ax.set_yscale("log")
    ax.set_title(f"ROC, m={mass} GeV")
    ax.grid(alpha=0.3)
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # 4) closure vs background rejection (Fig. 8 left style)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        arr("rejection_A"),
        arr("closure"),
        yerr=arr("closure_err"),
        fmt=".",
        markersize=4,
        elinewidth=0.7,
    )
    ax.axhline(1.0, color="k")
    ax.axhspan(
        1 - args.closure_tol,
        1 + args.closure_tol,
        color="g",
        alpha=0.15,
        label=f"+-{args.closure_tol:.0%}",
    )
    ax.set_xlabel("background rejection in A")
    ax.set_xscale("log")
    ax.set_ylabel("ABCD closure  (N_B N_C / N_D) / N_A  [bkg only]")
    ax.set_ylim(0.5, 1.5)
    ax.set_title(f"Closure vs rejection, m={mass} GeV")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # 5) r vs background rejection (Fig. 7 / 8 right / 12 style)
    fig, ax = plt.subplots(figsize=(8, 5))
    good = np.abs(arr("closure") - 1.0) < args.closure_tol
    ax.errorbar(
        arr("rejection_A"),
        arr("r"),
        yerr=arr("r_err"),
        fmt=".",
        markersize=4,
        elinewidth=0.7,
        color="0.7",
        label="all cuts",
    )
    ax.errorbar(
        arr("rejection_A")[good],
        arr("r")[good],
        yerr=arr("r_err")[good],
        fmt=".",
        markersize=5,
        elinewidth=0.7,
        color="b",
        label=f"closure within {args.closure_tol:.0%}",
    )
    ax.axhline(args.r_warn, color="r", linestyle="--", label=f"|r| = {args.r_warn}")
    ax.axhline(-args.r_warn, color="r", linestyle="--")
    ax.set_xlabel("background rejection in A")
    ax.set_xscale("log")
    ax.set_ylabel("normalized signal contamination r")
    ax.set_title(f"r vs rejection, m={mass} GeV")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # 6) contamination delta_i vs signal efficiency (Eq. 2.7)
    fig, ax = plt.subplots(figsize=(8, 5))
    for region, color in zip("ABCD", ("k", "b", "g", "m")):
        ax.plot(
            arr("eff_sig_A"),
            np.abs(arr(f"delta_{region}")),
            color=color,
            label=f"delta_{region}",
        )
    ax.axhline(
        args.delta_warn, color="r", linestyle="--", label=f"delta = {args.delta_warn}"
    )
    ax.set_xlabel("signal efficiency in A")
    ax.set_ylabel("|delta_i| = N_i,s / N_i,b")
    ax.set_yscale("log")
    ax.set_title(
        f"Signal contamination per region, m={mass} GeV"
        f" (signal scale {args.signal_scale})"
    )
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # 7) r and signal recovery vs signal efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        arr("eff_sig_A"),
        arr("r"),
        yerr=arr("r_err"),
        fmt=".-",
        markersize=4,
        elinewidth=0.7,
        label="r",
    )
    ax.plot(
        arr("eff_sig_A"),
        arr("signal_recovery"),
        ".-",
        markersize=4,
        label="signal recovery (N_A,a - N_A^pred,a) / N_A,s",
    )
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.axhline(1.0, color="k", linewidth=0.7)
    # recovery blows up where closure is broken: (1 - closure)/delta_A dominates
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("signal efficiency in A")
    ax.set_ylabel("value")
    ax.set_title(f"Contamination impact, m={mass} GeV")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)


def make_epoch_scan_outputs(mass, epoch_rows, output_folder, args):
    """Across-epoch outputs (paper Fig. 7 workflow): CSV + scatter/trend PDF.

    epoch_rows: one dict per (model label, eff target) working point, with
    keys from EPOCH_CSV_COLUMNS.
    """
    csv_path = os.path.join(output_folder, f"epoch_scan_m{mass}.csv")
    write_csv(epoch_rows, csv_path, EPOCH_CSV_COLUMNS)

    primary = [
        row for row in epoch_rows if abs(row["target_eff"] - args.primary_eff) < 1e-6
    ]
    with_epoch = sorted(
        (row for row in primary if row["epoch"] is not None),
        key=lambda row: row["epoch"],
    )
    finals = [row for row in primary if row["epoch"] is None]

    pdf_path = os.path.join(output_folder, f"epoch_scan_m{mass}.pdf")
    with PdfPages(pdf_path) as pdf:
        # 1) r vs rejection across epochs (Fig. 7 analog)
        fig, ax = plt.subplots(figsize=(8, 5))
        if with_epoch:
            epochs = np.array([row["epoch"] for row in with_epoch], dtype=float)
            scatter = ax.scatter(
                [row["rejection_A"] for row in with_epoch],
                [row["r"] for row in with_epoch],
                c=epochs,
                cmap="viridis",
                s=30,
                label="epoch checkpoints",
            )
            fig.colorbar(scatter, ax=ax, label="epoch")
            closure_pass = [
                row
                for row in with_epoch
                if abs(row["closure"] - 1.0) < args.closure_tol
            ]
            ax.scatter(
                [row["rejection_A"] for row in closure_pass],
                [row["r"] for row in closure_pass],
                facecolors="none",
                edgecolors="r",
                s=90,
                label=f"closure within {args.closure_tol:.0%}",
            )
        for row in finals:
            ax.scatter(
                [row["rejection_A"]],
                [row["r"]],
                marker="*",
                s=200,
                color="k",
                label=row["label"],
            )
        ax.axhline(args.r_warn, color="r", linestyle="--", label=f"|r| = {args.r_warn}")
        ax.set_xscale("log")
        ax.set_xlabel("background rejection in A")
        ax.set_ylabel("normalized signal contamination r")
        ax.set_title(
            f"Across-epoch scan, m={mass} GeV, eff_sig ~ {args.primary_eff:.0%}"
        )
        ax.legend(fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        # 2) closure / |r| / rejection vs epoch
        if with_epoch:
            epochs = [row["epoch"] for row in with_epoch]
            fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            axes[0].errorbar(
                epochs,
                [row["closure"] for row in with_epoch],
                yerr=[row["closure_err"] for row in with_epoch],
                fmt=".-",
            )
            axes[0].axhspan(
                1 - args.closure_tol, 1 + args.closure_tol, color="g", alpha=0.15
            )
            axes[0].axhline(1.0, color="k", linewidth=0.7)
            axes[0].set_ylabel("closure")
            axes[1].errorbar(
                epochs,
                [abs(row["r"]) for row in with_epoch],
                yerr=[row["r_err"] for row in with_epoch],
                fmt=".-",
            )
            axes[1].axhline(args.r_warn, color="r", linestyle="--")
            axes[1].set_ylabel("|r|")
            axes[2].plot(epochs, [row["rejection_A"] for row in with_epoch], ".-")
            axes[2].set_yscale("log")
            axes[2].set_ylabel("background rejection")
            axes[2].set_xlabel("epoch")
            for row in finals:
                for axis, value in zip(
                    axes, (row["closure"], abs(row["r"]), row["rejection_A"])
                ):
                    axis.axhline(value, color="k", linestyle=":", linewidth=0.9)
            axes[0].set_title(
                f"ABCD conditions vs epoch, m={mass} GeV,"
                f" eff_sig ~ {args.primary_eff:.0%}"
                + (f" (dotted: {finals[0]['label']})" if finals else "")
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"m={mass}: wrote {csv_path} and {pdf_path}")


# =========================================================
# DRIVER
# =========================================================
def analyze_model_mass(
    label,
    mass,
    score,
    mbb,
    in_window,
    in_sideband,
    is_signal,
    weights,
    output_folder,
    args,
    compute_dcorr,
    write_pdf,
):
    """Threshold scan + conditions for one (model, mass); write CSV (+PDF)."""
    rows = scan_thresholds(score, in_window, in_sideband, is_signal, weights, args)
    working_points = pick_working_points(rows, args.eff_targets)

    disco_result = None
    if compute_dcorr:
        acc_bkg = (in_window | in_sideband) & ~is_signal
        disco_result = disco_check(
            score,
            mbb,
            weights,
            acc_bkg,
            n_sub=args.dcorr_subsample,
            n_boot=args.dcorr_bootstrap,
        )

    primary_row = working_points.get(
        args.primary_eff, rows[len(rows) // 2] if rows else None
    )
    if primary_row is None:
        print(f"m={mass} [{label}]: empty threshold scan, skipping")
        return None
    flags = evaluate_flags(primary_row, disco_result, args)

    csv_path = os.path.join(output_folder, f"abcd_scan_{label}_m{mass}.csv")
    write_csv(rows, csv_path, CSV_COLUMNS)

    if write_pdf:
        pdf_path = os.path.join(output_folder, f"abcd_validation_{label}_m{mass}.pdf")
        with PdfPages(pdf_path) as pdf:
            summary_text_page(
                pdf, label, mass, working_points, disco_result, flags, args
            )
            make_plots(
                pdf,
                mass,
                rows,
                score,
                mbb,
                in_window,
                in_sideband,
                is_signal,
                weights,
                args,
            )
        print(f"m={mass} [{label}]: wrote {csv_path} and {pdf_path}")
    else:
        print(f"m={mass} [{label}]: wrote {csv_path}")

    summary = {
        "mass": int(mass),
        "label": label,
        "epoch": epoch_number(label),
        "disco": disco_result,
        "flags": flags,
        "working_points": {
            f"{target:.0%}": {
                key: (float(value) if np.isscalar(value) else value)
                for key, value in row.items()
            }
            for target, row in working_points.items()
        },
    }
    return summary, working_points


def validate_mass_point(
    mass,
    models,
    sessions,
    X,
    use_parametric,
    score_store,
    mbb,
    x_mass,
    class_target,
    weights,
    output_folder,
    args,
):
    """Validate every model at one mass point; returns per-model summaries."""
    is_signal_any = class_target == 0
    is_signal = is_signal_any & (x_mass == mass)
    is_background = ~is_signal_any
    considered = is_signal | is_background

    n_sig = int(is_signal.sum())
    if n_sig < args.min_signal_events:
        print(f"m={mass}: only {n_sig} signal events, skipping")
        return []

    in_window = (mbb >= args.mbb_low) & (mbb < args.mbb_high) & considered
    in_sideband = (mbb >= args.mbb_high) & (mbb < args.mbb_max) & considered
    if args.sideband == "both":
        in_sideband |= (mbb >= args.mbb_min) & (mbb < args.mbb_low) & considered

    acc_sig = (in_window | in_sideband) & is_signal
    acc_bkg = (in_window | in_sideband) & is_background
    print(
        f"m={mass}: {n_sig} signal events ({int(acc_sig.sum())} in acceptance), "
        f"{int(acc_bkg.sum())} background events in acceptance, "
        f"{len(models)} model(s) to validate"
    )

    if use_parametric:
        X[:, -1] = mass

    main_label = primary_label(models)
    summaries = []
    epoch_rows = []
    for label, model_path in models:
        if label not in sessions:
            sessions[label] = make_session(model_path)
        if use_parametric:
            score = run_inference(sessions[label], X, score_class=args.score_class)
        else:
            if label not in score_store:
                score_store[label] = run_inference(
                    sessions[label], X, score_class=args.score_class
                )
            score = score_store[label]

        result = analyze_model_mass(
            label,
            mass,
            score,
            mbb,
            in_window,
            in_sideband,
            is_signal,
            weights,
            output_folder,
            args,
            compute_dcorr=(args.dcorr_all_models or label == main_label),
            write_pdf=(args.full_pdfs or label == main_label),
        )
        if result is None:
            continue
        summary, working_points = result
        summaries.append(summary)
        for target, row in working_points.items():
            epoch_row = {key: row.get(key) for key in EPOCH_CSV_COLUMNS if key in row}
            epoch_row.update(label=label, epoch=epoch_number(label), target_eff=target)
            epoch_rows.append(epoch_row)

    if len(models) > 1 and epoch_rows:
        make_epoch_scan_outputs(mass, epoch_rows, output_folder, args)
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate the ABCDisCo (arXiv:2007.14400) assumptions for "
        "a trained DNN: background independence/closure, control-region "
        "contamination delta_i, and normalized signal contamination r."
    )
    parser.add_argument("--validation-file", required=True, type=str)
    parser.add_argument("--validation-weight-file", required=True, type=str)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-name", type=str, default=None, help="Single ONNX model file"
    )
    model_group.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Training output folder: validate every "
        "epoch_N.onnx checkpoint plus the final model",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        type=str,
        help="Model config yaml (training setup or dnn_config.yaml)",
    )
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--hme-friend-file", type=str, default=None)
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="With --model-dir: validate every Nth epoch "
        "checkpoint (the last one is always kept)",
    )
    parser.add_argument(
        "--full-pdfs",
        action="store_true",
        help="Write the full per-mass PDF for every model, " "not only the final one",
    )
    parser.add_argument(
        "--dcorr-all-models",
        action="store_true",
        help="Run the dCorr check for every model, not only " "the final one",
    )
    parser.add_argument(
        "--mass-points",
        type=int,
        nargs="*",
        default=None,
        help="Default: parametric_list masses present in the file",
    )
    parser.add_argument("--physics-weight-branch", type=str, default=None)
    parser.add_argument("--mbb-branch", type=str, default=DEFAULT_MBB_BRANCH)
    parser.add_argument("--mbb-low", type=float, default=70.0)
    parser.add_argument("--mbb-high", type=float, default=150.0)
    parser.add_argument("--mbb-min", type=float, default=70.0)
    parser.add_argument("--mbb-max", type=float, default=300.0)
    parser.add_argument(
        "--sideband",
        choices=["high", "both"],
        default="high",
        help="'high' matches the training (low sideband dropped)",
    )
    parser.add_argument("--score-class", type=int, default=0)
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=1.0,
        help="Extra scale on signal yields for delta_i (r is invariant)",
    )
    parser.add_argument("--n-scan", type=int, default=120)
    parser.add_argument(
        "--eff-targets", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.7]
    )
    parser.add_argument("--primary-eff", type=float, default=0.3)
    parser.add_argument("--closure-tol", type=float, default=0.10)
    parser.add_argument("--delta-warn", type=float, default=0.10)
    parser.add_argument("--r-warn", type=float, default=0.20)
    parser.add_argument("--dcorr-warn-ratio", type=float, default=5.0)
    parser.add_argument("--dcorr-abs-ok", type=float, default=0.01)
    parser.add_argument("--dcorr-subsample", type=int, default=4000)
    parser.add_argument("--dcorr-bootstrap", type=int, default=8)
    parser.add_argument("--min-signal-events", type=int, default=200)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    os.makedirs(args.output_folder, exist_ok=True)

    if args.model_dir is not None:
        models = discover_models(args.model_dir, epoch_step=args.epoch_step)
    else:
        models = [("best", args.model_name)]
    print(f"Models to validate: {[label for label, _ in models]}")

    with open(args.model_config, "r") as file:
        config = yaml.safe_load(file)

    X, mbb, x_mass, use_parametric = load_features(
        args.validation_file, config, hme_friend_file=args.hme_friend_file
    )
    class_target = load_weight_file(args.validation_weight_file)
    if len(class_target) != len(mbb):
        raise ValueError(
            f"Entry mismatch: weight file has {len(class_target)},"
            f" batch file has {len(mbb)}"
        )
    weights, weight_branch = load_physics_weight(
        args.validation_file, branch=args.physics_weight_branch
    )

    if args.mass_points:
        mass_points = args.mass_points
    else:
        parametric_list = get_config_value(config, "parametric_list", default=[])
        present = set(np.unique(x_mass[class_target == 0]).tolist())
        mass_points = [m for m in parametric_list if m in present]
    print(f"Mass points to validate: {mass_points}")

    sessions = {}
    score_store = {}
    summaries = []
    for mass in mass_points:
        summaries += validate_mass_point(
            mass,
            models,
            sessions,
            X,
            use_parametric,
            score_store,
            mbb,
            x_mass,
            class_target,
            weights,
            args.output_folder,
            args,
        )

    overview = {
        "inputs": {
            "validation_file": args.validation_file,
            "validation_weight_file": args.validation_weight_file,
            "models": {label: path for label, path in models},
            "physics_weight_branch": weight_branch,
            "mbb_branch": args.mbb_branch,
            "regions": {
                "window": [args.mbb_low, args.mbb_high],
                "sideband": [args.mbb_high, args.mbb_max],
                "sideband_mode": args.sideband,
            },
            "signal_scale": args.signal_scale,
        },
        "thresholds": {
            "closure_tol": args.closure_tol,
            "delta_warn": args.delta_warn,
            "r_warn": args.r_warn,
            "dcorr_warn_ratio": args.dcorr_warn_ratio,
        },
        "results": summaries,
    }
    summary_path = os.path.join(args.output_folder, "abcd_summary.yaml")
    with open(summary_path, "w") as file:
        yaml.dump(overview, file, sort_keys=False)
    print(f"Wrote {summary_path}")

    print("\n==== ABCD validation overview ====")
    main_label = primary_label(models)
    for summary in summaries:
        flags = summary["flags"]
        checked = [
            flags[key]
            for key in ("closure_ok", "delta_ok", "r_ok", "dcorr_ok")
            if key in flags
        ]
        status = "PASS" if all(checked) else "WARN"
        disco = summary["disco"]
        disco_text = (
            f"  (dCorr^2={disco['dcorr2']:.4f}, perm={disco['dcorr2_perm']:.4f})"
            if disco is not None
            else ""
        )
        marker = " <-- final" if summary["label"] == main_label else ""
        print(
            f"  m={summary['mass']:>5} {summary['label']:>10}  [{status}]  "
            f"closure_ok={flags['closure_ok']}  delta_ok={flags['delta_ok']}  "
            f"r_ok={flags['r_ok']}" + disco_text + marker
        )


if __name__ == "__main__":
    main()
