#!/usr/bin/env python3
import os
import argparse
import json
import yaml
import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim

# Import refactored helper functions, data loaders, and model architectures
from src.model_helper import (
    eval_parity_expr,
    get_class_setup,
    balance_weights,
    load_parametric_fold,
    load_physical_fold,
    score_hypotheses,
    DynamicMassManager,
    ParametricDNN,
    ONNXDeployWrapper,
    WeightedStandardCrossEntropyLoss,
)

try:
    import onnxscript  # noqa: F401

    ONNXSCRIPT_AVAILABLE = True
except ImportError:
    ONNXSCRIPT_AVAILABLE = False

# Use Apple Silicon MPS if available, otherwise GPU/CPU
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def save_loss_history(history, out_dir, regime):
    """Per-epoch train/test loss and learning rate: loss_history_{regime}.json and a plot."""
    with open(os.path.join(out_dir, f"loss_history_{regime}.json"), "w") as f:
        json.dump(history, f, indent=1)
    best = int(np.argmin(history["test_loss"]))
    fig, (ax, ax_lr) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["test_loss"], label="test (early stopping)")
    ax.axvline(history["epoch"][best], color="gray", ls="--", label="saved epoch")
    ax.set_ylabel("Weighted cross-entropy")
    ax.set_title(f"{regime.upper()} loss history")
    ax.legend()
    ax.grid(alpha=0.3)
    ax_lr.plot(history["epoch"], history["lr"], color="black")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curve_{regime}.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train Parametric DNNs using model_config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to YAML model configuration file",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Parity cycles to run (default: all), e.g. one per batch job",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="Regimes to train (default: the config's regimes)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the number of epochs"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads for torch; on shared nodes (lxplus) more threads run slower",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for torch and numpy (default: the config's 'seed', if any)",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    print(f"--> Loading configuration settings from: {args.config}")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg.get("seed")
    if seed is not None:
        # Network initialisation, epoch shuffling and background hypotheses; without a seed
        # every training is a different random draw
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"--> Random seed {seed}")

    # Resolve paths & target evaluation parameters
    input_folder = cfg["input_folder"]
    output_folder = cfg["output_folder"]
    tree_name = cfg.get("tree_name", "AppliedTree")
    dedicated_mass = cfg.get("dedicated_mass", None)

    # Class mapping & configuration
    class_setup = get_class_setup(cfg)
    if "output_classes" in cfg:
        class_names = class_setup["class_names"]
    else:
        class_names = cfg.get("class_names", ["Signal", "TT", "Other"])
    num_classes = len(class_names)
    weighting_cfg = cfg.get("weighting", {})
    flatten_signal_per_mass = weighting_cfg.get("flatten_signal_per_mass", False)
    flatten_min_events = weighting_cfg.get("flatten_min_events", 0)

    # ONNX Model Naming Pattern
    model_name_fmt = cfg.get("model_name", "pdnn_model_{regime}_nparity{nParity}.onnx")

    # Parity settings
    n_parity_folds = cfg.get("nParity", 4)
    train_parity_expr = cfg.get("train_parity").get("index")
    test_parity_expr = cfg.get("test_parity").get("index")
    val_parity_expr = cfg.get("val_parity").get("index")
    app_parity_expr = cfg.get("app_parity").get("index")

    if dedicated_mass is not None:
        eval_mass_points = [dedicated_mass]
        output_folder = f"{output_folder}_M{dedicated_mass}"
    else:
        eval_mass_points = cfg["signal_mass_points"]

    # Map dynamic regime feature vectors
    feature_map = {}
    for regime in cfg["regimes"]:
        feature_map[regime] = cfg["common_features"] + cfg["regime_features"].get(
            regime, []
        )

    extra_vars = cfg["extra_variables"]
    train_cfg = cfg["training"]

    # With 'spin' as an input, validation events are scored once per spin hypothesis
    spin_is_input = any("spin" in feats for feats in feature_map.values())
    eval_spins = class_setup["spin_values"] if spin_is_input else [None]

    print(f"Using compute acceleration device: {device}")
    print(f"Configured output classes ({num_classes}): {class_names}")
    regimes = args.regimes if args.regimes is not None else cfg["regimes"]
    folds = args.folds if args.folds is not None else range(n_parity_folds)

    # Loop over cross-validation parities defined in YAML configuration
    for i_fold in folds:
        # Dynamically evaluate target parity indices using string expressions from config
        train_parity = eval_parity_expr(train_parity_expr, i_fold)
        test_parity = eval_parity_expr(test_parity_expr, i_fold)
        val_parity = eval_parity_expr(val_parity_expr, i_fold)
        app_parity = eval_parity_expr(app_parity_expr, i_fold)

        out_dir = os.path.join(output_folder, f"nParity{train_parity}_validation")
        os.makedirs(out_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(
            f"RUNNING DUAL-REGIME PARITY CYCLE {i_fold} --> Train: {train_parity} | Test: {test_parity} | Val: {val_parity} | App: {app_parity}"
        )
        print("=" * 80 + "\n")

        files = {
            "train": {
                "data": os.path.join(
                    input_folder, f"nParity{train_parity}_Merged.root"
                ),
                "weight": os.path.join(
                    input_folder, f"nParity{train_parity}_Merged_weight.root"
                ),
            },
            "test": {
                "data": os.path.join(input_folder, f"nParity{test_parity}_Merged.root"),
                "weight": os.path.join(
                    input_folder, f"nParity{test_parity}_Merged_weight.root"
                ),
            },
            "val": {
                "data": os.path.join(input_folder, f"nParity{val_parity}_Merged.root"),
                "weight": os.path.join(
                    input_folder, f"nParity{val_parity}_Merged_weight.root"
                ),
            },
        }

        df_train_raw = load_parametric_fold(
            files["train"]["data"],
            files["train"]["weight"],
            feature_map,
            extra_vars,
            cfg["signal_mass_points"],
            cfg,
            seed=100,
            dedicated_mass=dedicated_mass,
        )
        df_test_raw = load_parametric_fold(
            files["test"]["data"],
            files["test"]["weight"],
            feature_map,
            extra_vars,
            cfg["signal_mass_points"],
            cfg,
            seed=200,
            dedicated_mass=dedicated_mass,
        )
        df_val_raw = load_physical_fold(
            files["val"]["data"],
            files["val"]["weight"],
            feature_map,
            extra_vars,
            cfg,
            dedicated_mass=dedicated_mass,
        )

        val_output_data = {
            "class_value": df_val_raw["class_value"].values.astype(np.int32),
            "weight_Central": df_val_raw["weight_Central"].values.astype(np.float32),
            "DeepHME_mass": df_val_raw["DeepHME_mass"].values.astype(np.float32),
            "res2b": df_val_raw["res2b"].values.astype(np.int32),
            "recovery": df_val_raw["recovery"].values.astype(np.int32),
            "boosted": df_val_raw["boosted"].values.astype(np.int32),
            "channelId": df_val_raw["channelId"].values.astype(np.int32),
            "X_mass": df_val_raw["X_mass"].values.astype(np.float32),
            "spin": df_val_raw["spin"].values.astype(np.int32),
        }

        for regime in regimes:
            print(f"\n--- Initiating Network Training for Regime: {regime.upper()} ---")
            current_features = feature_map[regime]

            val_mask = (
                (df_val_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_val_raw["boosted"] == 0)
            )
            train_mask = (
                (df_train_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_train_raw["boosted"] == 0)
            )
            test_mask = (
                (df_test_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_test_raw["boosted"] == 0)
            )

            df_train = (
                df_train_raw[train_mask]
                .copy()
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            df_test = (
                df_test_raw[test_mask]
                .copy()
                .sample(frac=1, random_state=200)
                .reset_index(drop=True)
            )

            if len(df_train) == 0 or len(df_test) == 0:
                print(
                    f"[WARNING] Insufficient statistics for regime {regime}. Skipping."
                )
                continue

            # Equal total weight per signal mass point (optional) and per output class
            for df_split in [df_train, df_test]:
                balance_weights(
                    df_split, num_classes, flatten_signal_per_mass, flatten_min_events
                )

            scaler = StandardScaler()
            scaler.fit(df_train[current_features])

            train_manager = DynamicMassManager(
                df_train,
                current_features,
                scaler,
                cfg["signal_mass_points"],
                spin_values=class_setup["spin_values"],
                is_training=True,
            )
            test_manager = DynamicMassManager(
                df_test,
                current_features,
                scaler,
                cfg["signal_mass_points"],
                spin_values=class_setup["spin_values"],
                is_training=False,
            )

            # Whole splits live on the device and batches are sliced there: a DataLoader
            # builds every batch on the CPU, which dominates the epoch time on a GPU
            X_test_dev, y_test_dev, w_test_dev = (
                t.to(device) for t in test_manager.get_scaled_tensors()
            )
            batch_size = train_cfg["batch_size"]
            eval_batch_size = train_cfg["eval_batch_size"]

            model = ParametricDNN(
                input_dim=len(current_features),
                num_classes=num_classes,
                hidden_dims=train_cfg.get("hidden_dims", [512, 256, 128, 64]),
                dropout_rate=train_cfg.get("dropout_rate", 0.35),
            ).to(device)

            criterion = WeightedStandardCrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )

            sched_cfg = train_cfg.get("lr_scheduler", {})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=sched_cfg.get("patience", 6),
                factor=sched_cfg.get("factor", 0.5),
            )

            best_loss = float("inf")
            epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
            early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
            epochs_no_improve = 0
            history = {"epoch": [], "train_loss": [], "test_loss": [], "lr": []}

            for epoch in range(1, epochs + 1):
                train_manager.randomize_background_masses()
                X_train_epoch, y_train_epoch, w_train_epoch = (
                    t.to(device) for t in train_manager.get_scaled_tensors()
                )

                model.train()
                train_loss = torch.zeros((), device=device)
                n_train = (len(X_train_epoch) // batch_size) * batch_size  # drop_last
                perm = torch.randperm(len(X_train_epoch), device=device)
                for start in range(0, n_train, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_y = y_train_epoch[idx]
                    optimizer.zero_grad()

                    outputs = model(X_train_epoch[idx])
                    loss = criterion(outputs, batch_y, w_train_epoch[idx])
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.detach() * len(batch_y)

                train_loss = train_loss.item() / n_train

                model.eval()
                val_loss = torch.zeros((), device=device)
                with torch.no_grad():
                    for start in range(0, len(X_test_dev), eval_batch_size):
                        sl = slice(start, start + eval_batch_size)
                        outputs = model(X_test_dev[sl])
                        loss = criterion(outputs, y_test_dev[sl], w_test_dev[sl])
                        val_loss += loss * len(y_test_dev[sl])
                val_loss = val_loss.item() / len(X_test_dev)

                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["test_loss"].append(val_loss)
                history["lr"].append(optimizer.param_groups[0]["lr"])
                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(out_dir, f"best_pdnn_model_{regime}.pth"),
                    )
                else:
                    epochs_no_improve += 1

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"    [{regime.upper()}] Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.5f} | Test Loss: {val_loss:.5f}"
                    )

                if epochs_no_improve >= early_stopping_patience:
                    print(
                        f"    [{regime.upper()}] Early stopping triggered at epoch {epoch}. Restoring best weights."
                    )
                    break

            save_loss_history(history, out_dir, regime)
            model.load_state_dict(
                torch.load(os.path.join(out_dir, f"best_pdnn_model_{regime}.pth"))
            )
            model.eval()

            # Export ONNX model with embedded standard scaler dynamically formatted
            onnx_wrapper = ONNXDeployWrapper(model.to("cpu"), scaler)
            onnx_wrapper.eval()

            dummy_input = torch.randn(1, len(current_features), dtype=torch.float32)
            onnx_filename = model_name_fmt.format(nParity=i_fold, regime=regime)
            onnx_path = os.path.join(out_dir, onnx_filename)

            torch.onnx.export(
                onnx_wrapper,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["raw_inputs"],
                output_names=["probabilities"],
                dynamic_axes={
                    "raw_inputs": {0: "batch_size"},
                    "probabilities": {0: "batch_size"},
                },
                # The dynamo exporter needs onnxscript; fall back to the TorchScript one without it
                dynamo=ONNXSCRIPT_AVAILABLE,
            )
            print(
                f"    [{regime.upper()}] Successfully exported ONNX model to: {onnx_path}"
            )

            model = model.to(device)
            model.eval()

            val_indices = np.where(val_mask)[0]
            df_val_temp = df_val_raw[val_mask].copy()

            hypotheses = [(m, sp) for m in eval_mass_points for sp in eval_spins]
            for mass, spin, outputs, logit_score in score_hypotheses(
                model, scaler, df_val_temp, current_features, hypotheses, device
            ):
                score_name = f"DNN_M{mass}" if spin is None else f"DNN_M{mass}_S{spin}"

                # Dynamic branch generation based on class_names
                targets_to_save = [(score_name, logit_score)]
                for c_idx, c_name in enumerate(class_names):
                    targets_to_save.append(
                        (f"{score_name}_prob_{c_name.lower()}", outputs[:, c_idx])
                    )

                for key, source_arr in targets_to_save:
                    if key not in val_output_data:
                        val_output_data[key] = np.zeros(
                            len(df_val_raw), dtype=np.float32
                        )
                    val_output_data[key][val_indices] = source_arr.astype(np.float32)

        # Output ROOT File Construction
        output_root_filename = os.path.join(out_dir, "validation_applied.root")
        print(f"\n--> Saving applied dataset tree to '{output_root_filename}'...")
        with uproot.recreate(output_root_filename) as f_out:
            f_out.mktree(tree_name, {k: v.dtype for k, v in val_output_data.items()})
            f_out[tree_name].extend(val_output_data)

    print("\n--> Training execution complete.")


if __name__ == "__main__":
    main()
