#!/usr/bin/env python3
"""
Overtraining check for a trained pDNN.

For every parity cycle and regime, the exported ONNX model is evaluated on the parity it was
trained on and on its validation parity (never seen in training or early stopping), at a few
mass (and spin) hypotheses. For each output class the signal-logit distributions of the two
are compared with a weighted Kolmogorov-Smirnov test, and the signal-vs-background ROC AUC is
computed on both. A model that learned its training events scores them visibly better than
the validation events: AUC(train) > AUC(val) and small KS p-values.

Usage:
    python3 overtraining_check.py --config model_configs/v2610_pDNN_v1.yaml
"""

import os
import json
import argparse
import yaml
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score

from src.model_helper import (
    eval_parity_expr,
    get_class_setup,
    load_physical_fold,
    update_mass_dependent_features,
)


def weighted_ks(x1, w1, x2, w2):
    """Weighted two-sample KS distance, and its p-value with Kish effective sample sizes."""
    x = np.concatenate([x1, x2])
    order = np.argsort(x)
    in1 = np.concatenate([np.ones(len(x1), bool), np.zeros(len(x2), bool)])[order]
    w = np.concatenate([w1 / w1.sum(), w2 / w2.sum()])[order]
    cdf1 = np.cumsum(np.where(in1, w, 0.0))
    cdf2 = np.cumsum(np.where(in1, 0.0, w))
    d = np.max(np.abs(cdf1 - cdf2))
    n1 = w1.sum() ** 2 / (w1**2).sum()
    n2 = w2.sum() ** 2 / (w2**2).sum()
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0, 1))
    return float(d), p


def score(session, df, features, mass, spin):
    df = df.copy()
    df["X_mass"] = mass
    if spin is not None:
        df["spin"] = spin
    df = update_mass_dependent_features(df, features)
    x = df[features].values.astype(np.float32)
    name_in = session.get_inputs()[0].name
    probs = np.concatenate(
        [
            session.run(None, {name_in: x[i : i + 65536]})[0]
            for i in range(0, len(x), 65536)
        ]
    )
    # float64: in float32, 1 - 1e-15 rounds to 1 and the logit of very signal-like events is inf
    p = np.clip(probs[:, 0].astype(np.float64), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def main():
    parser = argparse.ArgumentParser(
        description="Train vs validation overtraining check"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--masses", type=int, nargs="+", default=[300, 500, 800, 1000, 1500]
    )
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--bkg-fraction", type=float, default=0.25, help="Background subsample"
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    class_setup = get_class_setup(cfg)
    class_names = (
        class_setup["class_names"]
        if "output_classes" in cfg
        else cfg.get("class_names", ["Signal", "TT", "Other"])
    )
    feature_map = {
        r: cfg["common_features"] + cfg["regime_features"].get(r, [])
        for r in cfg["regimes"]
    }
    spin_is_input = any("spin" in f for f in feature_map.values())
    spins = class_setup["spin_values"] if spin_is_input else [None]
    model_name_fmt = cfg.get("model_name", "pdnn_model_{regime}_nparity{nParity}.onnx")
    out_dir = args.output_dir or os.path.join(cfg["output_folder"], "overtraining")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    results = []
    folds = args.folds if args.folds is not None else range(cfg.get("nParity", 4))
    with PdfPages(os.path.join(out_dir, "overtraining_check.pdf")) as pdf:
        for i_fold in folds:
            train_parity = eval_parity_expr(cfg["train_parity"]["index"], i_fold)
            val_parity = eval_parity_expr(cfg["val_parity"]["index"], i_fold)
            fold_dir = os.path.join(
                cfg["output_folder"], f"nParity{train_parity}_validation"
            )
            samples = {}
            for label, parity in [("train", train_parity), ("val", val_parity)]:
                path = os.path.join(cfg["input_folder"], f"nParity{parity}_Merged.root")
                df = load_physical_fold(
                    path, None, feature_map, cfg["extra_variables"], cfg
                )
                df = df[df["class_targets_multiclass"] >= 0]
                keep = (df["class_targets_multiclass"].values == 0) | (
                    rng.random(len(df)) < args.bkg_fraction
                )
                samples[label] = df[keep].reset_index(drop=True)
            print(
                f"cycle {i_fold}: train parity {train_parity}, val parity {val_parity}"
            )

            for regime in cfg["regimes"]:
                model_path = os.path.join(
                    fold_dir, model_name_fmt.format(nParity=i_fold, regime=regime)
                )
                if not os.path.exists(model_path):
                    print(f"  [WARNING] missing {model_path}")
                    continue
                session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                features = feature_map[regime]
                for mass in args.masses:
                    for spin in spins:
                        scored = {}
                        for label, df in samples.items():
                            in_regime = (
                                (df["boosted"] == 1)
                                if regime == "boosted"
                                else (df["boosted"] == 0)
                            )
                            d = df[in_regime]
                            y = d["class_targets_multiclass"].values
                            sig = (y == 0) & (d["X_mass"].values == mass)
                            if spin is not None:
                                sig &= d["spin"].values == spin
                            sel = sig | (y > 0)
                            d = d[sel]
                            scored[label] = (
                                score(session, d, features, mass, spin),
                                np.where(sig[sel], 0, y[sel]),
                                np.clip(
                                    d["weight_Central"].values.astype(np.float64),
                                    0,
                                    None,
                                ),
                            )
                        row = {
                            "cycle": i_fold,
                            "regime": regime,
                            "mass": mass,
                            "spin": spin,
                        }
                        fig, axes = plt.subplots(
                            1, len(class_names), figsize=(4 * len(class_names), 3.5)
                        )
                        for c, cname in enumerate(class_names):
                            (s_tr, y_tr, w_tr), (s_va, y_va, w_va) = (
                                scored["train"],
                                scored["val"],
                            )
                            a, b = y_tr == c, y_va == c
                            if (
                                a.sum() < 20
                                or b.sum() < 20
                                or w_tr[a].sum() <= 0
                                or w_va[b].sum() <= 0
                            ):
                                continue
                            d_ks, p_ks = weighted_ks(s_tr[a], w_tr[a], s_va[b], w_va[b])
                            row[f"ks_{cname}"] = d_ks
                            row[f"ksp_{cname}"] = p_ks
                            edges = np.linspace(
                                min(s_tr[a].min(), s_va[b].min()),
                                max(s_tr[a].max(), s_va[b].max()),
                                41,
                            )
                            ax = axes[c]
                            ax.hist(
                                s_tr[a],
                                bins=edges,
                                weights=w_tr[a] / w_tr[a].sum(),
                                alpha=0.4,
                                label="train",
                            )
                            h, _ = np.histogram(
                                s_va[b], bins=edges, weights=w_va[b] / w_va[b].sum()
                            )
                            ax.plot(
                                0.5 * (edges[1:] + edges[:-1]),
                                h,
                                "k.",
                                label="validation",
                            )
                            ax.set_title(f"{cname}: KS p = {p_ks:.2g}")
                            ax.set_yscale("log")
                            ax.set_xlabel("signal logit")
                            ax.legend(fontsize=8)
                        for label in ("train", "val"):
                            s_, y_, w_ = scored[label]
                            sw = np.where(
                                y_ == 0,
                                w_ / max(w_[y_ == 0].sum(), 1e-12),
                                w_ / max(w_[y_ > 0].sum(), 1e-12),
                            )
                            if (y_ == 0).sum() > 0 and (y_ > 0).sum() > 0:
                                row[f"auc_{label}"] = float(
                                    roc_auc_score(y_ == 0, s_, sample_weight=sw)
                                )
                        fig.suptitle(
                            f"cycle {i_fold} {regime} M{mass}"
                            + (f" spin {spin}" if spin is not None else "")
                            + f" | AUC train {row.get('auc_train', np.nan):.4f} val {row.get('auc_val', np.nan):.4f}"
                        )
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        results.append(row)
                        print(
                            f"  {regime:8s} M{mass:<5d}"
                            + (f" S{spin}" if spin is not None else "")
                            + f"  AUC train {row.get('auc_train', np.nan):.4f}  val {row.get('auc_val', np.nan):.4f}  "
                            + "  ".join(
                                f"KS-p {c} {row.get('ksp_' + c, np.nan):.2g}"
                                for c in class_names
                            )
                        )

    with open(os.path.join(out_dir, "overtraining_check.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"--> Wrote {out_dir}/overtraining_check.pdf and .json")


if __name__ == "__main__":
    main()
