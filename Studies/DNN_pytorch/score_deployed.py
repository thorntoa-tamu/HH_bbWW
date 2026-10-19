#!/usr/bin/env python3
"""
Scores a training dataset with a deployed pDNN (config/DNN/<version>, the analysis' ONNX models and
dnn_config.yaml with integer fold offsets), the way Analysis/DNN_Application.py applies it: the
model of fold f scores the events with event % nParity == (f + app offset) % nParity. Writes one
tree per dataset parity in the validation_applied.root layout, so limit_pdnn.py can compare the
deployed model with newly trained ones on the same events.

Usage:
    python3 score_deployed.py --model-dirs ../../config/DNN/DoubleLepton_Parametric_Resolved_v1 \
        ../../config/DNN/DoubleLepton_Parametric_Boosted_v1 --config model_configs/<limit config>.yaml \
        --parities 0 1 2 3
The --config supplies input_folder, output_folder, signal_mass_points and the class setup.
"""

import os
import argparse
import yaml
import uproot
import numpy as np
import onnxruntime as ort

from src.model_helper import load_physical_fold


def main():
    parser = argparse.ArgumentParser(description="Score a dataset with a deployed pDNN")
    parser.add_argument(
        "--model-dirs", nargs="+", required=True, help="config/DNN/<version> per regime"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--parities", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    models = {}
    for d in args.model_dirs:
        with open(os.path.join(d, "dnn_config.yaml")) as f:
            mcfg = yaml.safe_load(f)
        regime = (
            "boosted"
            if "boosted" in os.path.basename(d.rstrip("/")).lower()
            else "resolved"
        )
        models[regime] = (d, mcfg)
    feature_map = {
        r: m["common_features"] + m["regime_features"][r]
        for r, (_, m) in models.items()
    }
    for r in ("resolved", "boosted"):
        feature_map.setdefault(r, [])
    extra = sorted(
        set(sum((m.get("extra_variables", []) for _, m in models.values()), []))
        | {
            "channelId",
            "class_value",
            "DeepHME_mass",
            "recovery",
            "boosted",
            "res2b",
            "event",
        }
    )
    masses = cfg["signal_mass_points"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads

    for parity in args.parities:
        path = os.path.join(cfg["input_folder"], f"nParity{parity}_Merged.root")
        df = load_physical_fold(path, None, feature_map, extra, cfg)
        out = {
            "class_value": df["class_value"].values.astype(np.int32),
            "weight_Central": df["weight_Central"].values.astype(np.float32),
            "DeepHME_mass": df["DeepHME_mass"].values.astype(np.float32),
            "res2b": df["res2b"].values.astype(np.int32),
            "recovery": df["recovery"].values.astype(np.int32),
            "boosted": df["boosted"].values.astype(np.int32),
            "channelId": df["channelId"].values.astype(np.int32),
            "X_mass": df["X_mass"].values.astype(np.float32),
            "spin": df["spin"].values.astype(np.int32),
        }
        for regime, (d, mcfg) in models.items():
            n = mcfg.get("nParity", 4)
            fold = (parity - mcfg["app_parity"]["offset"]) % n
            name = mcfg["model_name"].format(fold=fold, regime=regime, nParity=fold)
            session = ort.InferenceSession(
                os.path.join(d, name), opts, providers=["CPUExecutionProvider"]
            )
            in_name = session.get_inputs()[0].name
            mask = (
                (df["boosted"].values == 1)
                if regime == "boosted"
                else (df["boosted"].values == 0)
            )
            idx = np.nonzero(mask)[0]
            features = feature_map[regime]
            base = {}
            for f in features:
                if f.endswith("_scaled"):
                    base[f] = (
                        df[f.replace("_scaled", "")].values[idx].astype(np.float32)
                    )
                elif f != "X_mass":
                    base[f] = df[f].values[idx].astype(np.float32)
            print(
                f"parity {parity}: {regime} fold {fold} ({name}), {len(idx)} events",
                flush=True,
            )
            for mass in masses:
                X = np.stack(
                    [
                        (
                            np.full(len(idx), mass, np.float32)
                            if f == "X_mass"
                            else base[f] * mass if f.endswith("_scaled") else base[f]
                        )
                        for f in features
                    ],
                    axis=1,
                )
                probs = np.concatenate(
                    [
                        session.run(None, {in_name: X[i : i + 65536]})[0]
                        for i in range(0, len(X), 65536)
                    ]
                ).astype(np.float64)
                p = np.clip(probs[:, 0], 1e-15, 1 - 1e-15)
                key = f"DNN_M{mass}"
                if key not in out:
                    out[key] = np.zeros(len(df), np.float32)
                out[key][idx] = (np.log(p) - np.log1p(-p)).astype(np.float32)
        # One file per parity; limit_pdnn.py reads nParity{k}_validation for k = 0..3
        out_dir = os.path.join(cfg["output_folder"], f"nParity{parity}_validation")
        os.makedirs(out_dir, exist_ok=True)
        with uproot.recreate(os.path.join(out_dir, "validation_applied.root")) as f:
            f.mktree(
                cfg.get("tree_name", "AppliedTree"),
                {k: v.dtype for k, v in out.items()},
            )
            f[cfg.get("tree_name", "AppliedTree")].extend(out)
        print(f"parity {parity}: written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
