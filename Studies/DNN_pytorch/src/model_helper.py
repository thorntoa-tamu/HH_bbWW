#!/usr/bin/env python3
import uproot
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 1. Parity Expression & Feature Calculation Helpers
# ==============================================================================
def eval_parity_expr(expr_str, n_parity_val):
    """
    Evaluates parity string expressions from configuration file (e.g. '(nParity + 1) % 4')
    by injecting 'nParity' into context.
    """
    return eval(expr_str, {"nParity": n_parity_val, "np": np})


def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)


def compute_static_features(df):
    safe_met = np.clip(df["PuppiMET_pt"], 1e-3, None)
    safe_ht = np.clip(df["HT"], 1.0, None)

    dr_l1_b1 = delta_r(df["lep1_eta"], df["lep1_phi"], df["bjet1_eta"], df["bjet1_phi"])
    dr_l1_b2 = delta_r(df["lep1_eta"], df["lep1_phi"], df["bjet2_eta"], df["bjet2_phi"])
    dr_l2_b1 = delta_r(df["lep2_eta"], df["lep2_phi"], df["bjet1_eta"], df["bjet1_phi"])
    dr_l2_b2 = delta_r(df["lep2_eta"], df["lep2_phi"], df["bjet2_eta"], df["bjet2_phi"])

    df["min_dR_l_b"] = np.minimum(
        np.minimum(dr_l1_b1, dr_l1_b2), np.minimum(dr_l2_b1, dr_l2_b2)
    )
    df["lep_b_centrality"] = (df["lep1_pt"] + df["lep2_pt"]) / np.clip(
        df["bjet1_pt"] + df["bjet2_pt"] + df["PuppiMET_pt"], 1.0, None
    )
    df["MET_over_sqrt_HT"] = df["PuppiMET_pt"] / np.sqrt(safe_ht)
    df["HT_total"] = df["HT"] + df["PuppiMET_pt"] + df["lep1_pt"] + df["lep2_pt"]

    if "bjet1_btagPNetB" in df.columns:
        df["bb_btag_product"] = df["bjet1_btagPNetB"] * df["bjet2_btagPNetB"]
    if "bb_pt" in df.columns:
        df["pt_ratio_ll_bb"] = (df["ll_pt"]) / np.clip(df["bb_pt"], 1.0, None)

    mt_l1_met = np.sqrt(
        2 * df["lep1_pt"] * safe_met * (1 - np.cos(df["lep1_phi"] - df["PuppiMET_phi"]))
    )
    mt_l2_met = np.sqrt(
        2 * df["lep2_pt"] * safe_met * (1 - np.cos(df["lep2_phi"] - df["PuppiMET_phi"]))
    )
    df["min_MT_lep_met"] = np.minimum(mt_l1_met, mt_l2_met)

    df["cos_theta_star_ll"] = np.tanh((df["lep1_eta"] - df["lep2_eta"]) / 2.0)

    dphi_l1_met = np.abs(
        (df["lep1_phi"] - df["PuppiMET_phi"] + np.pi) % (2 * np.pi) - np.pi
    )
    dphi_l2_met = np.abs(
        (df["lep2_phi"] - df["PuppiMET_phi"] + np.pi) % (2 * np.pi) - np.pi
    )
    df["min_dphi_l_met"] = np.minimum(dphi_l1_met, dphi_l2_met)

    px_ll = df["lep1_pt"] * np.cos(df["lep1_phi"]) + df["lep2_pt"] * np.cos(
        df["lep2_phi"]
    )
    py_ll = df["lep1_pt"] * np.sin(df["lep1_phi"]) + df["lep2_pt"] * np.sin(
        df["lep2_phi"]
    )

    pt_ll = np.sqrt(px_ll**2 + py_ll**2)
    m_ll = df["ll_mass"]
    e_ll = np.sqrt(pt_ll**2 + m_ll**2)

    df["m_ll_met_transverse"] = np.sqrt(
        np.clip(
            (e_ll + safe_met) ** 2
            - (px_ll + safe_met * np.cos(df["PuppiMET_phi"])) ** 2
            - (py_ll + safe_met * np.sin(df["PuppiMET_phi"])) ** 2,
            0.0,
            None,
        )
    )

    dphi_leptons = df["lep1_phi"] - df["lep2_phi"]
    df["ll_dphi"] = np.abs((dphi_leptons + np.pi) % (2 * np.pi) - np.pi)
    phi_ll = np.arctan2(py_ll, px_ll)
    df["met_ll_dphi"] = np.abs(
        (phi_ll - df["PuppiMET_phi"] + np.pi) % (2 * np.pi) - np.pi
    )
    return df


def update_mass_dependent_features(df, features):
    """
    Dynamically creates scaled features for parametric inference.
    If a required feature contains '_scaled', its base counterpart is multiplied by mX.
    """
    mX = df["X_mass"].values
    for feature in features:
        if "_scaled" in feature:
            base_feature = feature.replace("_scaled", "")
            if base_feature in df.columns:
                df[feature] = df[base_feature].values * mX
            else:
                print(
                    f"[WARNING] Base feature '{base_feature}' for '{feature}' not found in DataFrame!"
                )
    return df


# ==============================================================================
# 2. Class, Weight & Data Loader Functions
# ==============================================================================
def get_class_setup(cfg):
    """
    Resolves which dataset class_values are signal and how class_values map onto the
    network output nodes.

    signal_classes: class_values treated as signal. Default: every class_value <= 0.
    output_classes: ordered {node name: "signal" | [class_values] | "rest"}. The first node
        must be "signal"; "rest" collects every background class_value not listed elsewhere.
        Default: Signal / TT (class_value 1) / Other (rest), the original pDNN classes.
    Events whose class_value maps to no node (e.g. unused signals) get target -1.
    """
    output_classes = cfg.get("output_classes")
    if output_classes is None:
        output_classes = {"Signal": "signal", "TT": [1], "Other": "rest"}
    names = list(output_classes.keys())
    if output_classes[names[0]] != "signal":
        raise ValueError("The first entry of output_classes must be 'signal'.")
    return {
        "signal_classes": cfg.get("signal_classes"),
        "output_classes": output_classes,
        "class_names": names,
        "signal_spin": cfg.get("signal_spin", {}),
        "spin_values": cfg.get("spin_values", [0]),
    }


def is_signal(class_value, class_setup):
    signal_classes = class_setup["signal_classes"]
    if signal_classes is None:
        return class_value <= 0
    return np.isin(class_value, signal_classes)


def assign_targets(class_value, class_setup):
    class_value = np.asarray(class_value)
    targets = np.full(len(class_value), -1, dtype=np.int64)
    sig = is_signal(class_value, class_setup)
    listed = []
    rest_node = None
    for node, (name, spec) in enumerate(class_setup["output_classes"].items()):
        if spec == "signal":
            targets[sig] = node
        elif spec == "rest":
            rest_node = node
        else:
            targets[np.isin(class_value, spec) & ~sig] = node
            listed += list(spec)
    if rest_node is not None:
        targets[(class_value > 0) & ~np.isin(class_value, listed) & ~sig] = rest_node
    return targets


def compute_base_weights(weight_central, weighting_cfg):
    """Event weights from weight_Central: negative weights to 0, then clip at mean + N sigma."""
    w = np.asarray(weight_central, dtype=np.float64).copy()
    if weighting_cfg.get("zero_negative", True):
        w = np.where(w <= 0, 0.0, w)
    clip_sigma = weighting_cfg.get("clip_sigma", 3.0)
    if clip_sigma is not None:
        w = np.clip(w, None, np.mean(np.abs(w)) + clip_sigma * np.std(np.abs(w)))
    return w


def balance_weights(df, num_classes, flatten_signal_per_mass, flatten_min_events=0):
    """
    Rescales 'class_weight' in place for one regime and split: optionally gives every signal
    mass point the same total weight, then gives every output class the total of the signal.

    Mass points with fewer than flatten_min_events signal events keep their own weight instead
    of being scaled up (e.g. low masses in the boosted regime, with only a few events), and the
    remaining signal weight is shared equally by the other mass points.
    """
    y_arr = df["class_targets_multiclass"].values
    if flatten_signal_per_mass:
        sig = y_arr == 0
        x_mass = df["X_mass"].values
        masses, counts = np.unique(x_mass[sig], return_counts=True)
        flat_masses = masses[counts >= flatten_min_events]
        kept = sig & ~np.isin(x_mass, flat_masses)
        flat_total = (
            df.loc[sig, "class_weight"].sum() - df.loc[kept, "class_weight"].sum()
        )
        for mass in flat_masses:
            mask = sig & (x_mass == mass)
            mass_total = df.loc[mask, "class_weight"].sum()
            if mass_total > 0:
                df.loc[mask, "class_weight"] *= (
                    flat_total / len(flat_masses) / mass_total
                )

    sums = {c: df.loc[y_arr == c, "class_weight"].sum() for c in range(num_classes)}
    target_sum = sums.get(0, 1.0)  # Signal class index 0 as target baseline
    if target_sum <= 0:
        target_sum = 1.0
    for c in range(num_classes):
        if sums[c] > 0:
            df.loc[y_arr == c, "class_weight"] *= target_sum / sums[c]


def add_derived_features(df, features):
    if "DeepHME_mass_error_rel" in features:
        df["DeepHME_mass_error_rel"] = df["DeepHME_mass_error"] / np.clip(
            df["DeepHME_mass"], 1.0, None
        )
    return df


def _load_fold(data_path, weight_path, feature_map, extra_vars, cfg):
    class_setup = get_class_setup(cfg)
    all_possible_features = list(set(feature_map["resolved"] + feature_map["boosted"]))

    # Exclude dynamic '_scaled', derived and spin features from ROOT branch loading
    parent_features = [
        f
        for f in all_possible_features
        if "_scaled" not in f and f not in ("spin", "DeepHME_mass_error_rel")
    ]
    if "DeepHME_mass_error_rel" in all_possible_features:
        parent_features += ["DeepHME_mass_error", "DeepHME_mass"]
    load_branches = list(
        set(parent_features + extra_vars + ["weight_Central", "class_value", "X_mass"])
    )

    with uproot.open(data_path) as f_data:
        available = [b for b in load_branches if b in f_data["Events"]]
        df = f_data["Events"].arrays(available, library="pd")

    if "weighting" in cfg:
        df["class_weight"] = compute_base_weights(
            df["weight_Central"], cfg["weighting"]
        )
    else:
        # Legacy datasets: weights precomputed by create_dataset.py
        with uproot.open(weight_path) as f_weight:
            df["class_weight"] = f_weight["weight_tree"]["class_weight"].array(
                library="np"
            )

    class_value = df["class_value"].values
    df["class_targets_multiclass"] = assign_targets(class_value, class_setup)
    df["class_targets_binary"] = is_signal(class_value, class_setup).astype(np.int64)
    df["spin"] = np.zeros(len(df), dtype=np.int32)
    for c, spin in class_setup["signal_spin"].items():
        df.loc[class_value == int(c), "spin"] = int(spin)
    df = add_derived_features(df, all_possible_features)

    float_cols = [
        c for c in df.columns if df[c].dtype == np.float64 and c != "class_weight"
    ]
    df[float_cols] = df[float_cols].astype(np.float32)
    return df, all_possible_features, class_setup


def load_parametric_fold(
    data_path,
    weight_path,
    feature_map,
    extra_vars,
    mass_points,
    cfg,
    seed=42,
    dedicated_mass=None,
):
    """Training/test fold: only events with an output class; backgrounds get random hypotheses."""
    df, all_possible_features, class_setup = _load_fold(
        data_path, weight_path, feature_map, extra_vars, cfg
    )
    df = df[df["class_targets_multiclass"] >= 0]

    is_bkg = df["class_targets_binary"] == 0
    df_sig = df[~is_bkg].copy()
    df_bkg = df[is_bkg].copy()

    if dedicated_mass is not None:
        df_sig = df_sig[df_sig["X_mass"] == dedicated_mass].copy()
        df_bkg["X_mass"] = dedicated_mass
    else:
        rng = np.random.default_rng(seed)
        df_bkg["X_mass"] = rng.choice(mass_points, size=len(df_bkg)).astype(np.int32)
        df_bkg["spin"] = rng.choice(
            class_setup["spin_values"], size=len(df_bkg)
        ).astype(np.int32)

    df_final = pd.concat([df_sig, df_bkg], ignore_index=True)
    df_final["X_mass"] = df_final["X_mass"].astype(np.int32)
    df_final = update_mass_dependent_features(df_final, all_possible_features)
    return df_final


def load_physical_fold(
    data_path, weight_path, feature_map, extra_vars, cfg, dedicated_mass=None
):
    """Validation fold: every event, with its physical X_mass (0 for backgrounds)."""
    df, all_possible_features, class_setup = _load_fold(
        data_path, weight_path, feature_map, extra_vars, cfg
    )
    if dedicated_mass is not None:
        is_bkg = df["class_targets_binary"] == 0
        is_target_sig = (df["class_targets_binary"] == 1) & (
            df["X_mass"] == dedicated_mass
        )
        df = df[is_bkg | is_target_sig].copy()
    df = update_mass_dependent_features(df, all_possible_features)
    return df.reset_index(drop=True)


# ==============================================================================
# 3. Dynamic Mass State Manager & PyTorch Network Classes
# ==============================================================================
class DynamicMassManager:
    def __init__(
        self, df, features, scaler, mass_points, spin_values=(0,), is_training=True
    ):
        self.df = df.copy().reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.mass_points = mass_points
        self.spin_values = list(spin_values)
        self.is_training = is_training

        self.bkg_mask = (self.df["class_targets_multiclass"] != 0).values
        self.num_bkg = self.bkg_mask.sum()

    def randomize_background_masses(self):
        if self.is_training and self.num_bkg > 0:
            new_masses = np.random.choice(self.mass_points, size=self.num_bkg)
            self.df.loc[self.bkg_mask, "X_mass"] = new_masses.astype(np.int32)
            if "spin" in self.features:
                new_spins = np.random.choice(self.spin_values, size=self.num_bkg)
                self.df.loc[self.bkg_mask, "spin"] = new_spins.astype(np.int32)
            self.df = update_mass_dependent_features(self.df, self.features)

    def get_scaled_tensors(self):
        X_scaled = self.scaler.transform(self.df[self.features])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(
            self.df["class_targets_multiclass"].values, dtype=torch.long
        )
        w_tensor = torch.tensor(self.df["class_weight"].values, dtype=torch.float32)
        return X_tensor, y_tensor, w_tensor


def score_hypotheses(model, scaler, df, features, hypotheses, device, batch_size=65536):
    """
    Yields (mass, spin, class probabilities, signal log-odds) of every event in df for each
    (mass, spin) hypothesis (spin None: not an input). The log-odds log(p_sig / (1 - p_sig)) is
    computed from the logits, so it does not saturate where p_sig rounds to 1 in float32. The inputs are built on the device once; per
    hypothesis only X_mass, spin and the mass-scaled features are refilled, so scoring is
    not limited by per-event batching on the CPU.
    """
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    columns = []
    for feature in features:
        if feature in ("X_mass", "spin"):
            columns.append(np.zeros(len(df), dtype=np.float32))
        elif feature.endswith("_scaled"):
            columns.append(df[feature.replace("_scaled", "")].values.astype(np.float32))
        else:
            columns.append(df[feature].values.astype(np.float32))
    X = torch.tensor(np.stack(columns, axis=1), device=device)
    scaled_bases = {
        i: X[:, i].clone() for i, f in enumerate(features) if f.endswith("_scaled")
    }
    for mass, spin in hypotheses:
        for i, feature in enumerate(features):
            if feature == "X_mass":
                X[:, i] = mass
            elif feature == "spin" and spin is not None:
                X[:, i] = spin
            elif i in scaled_bases:
                X[:, i] = scaled_bases[i] * mass
        probs, log_odds = [], []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                logits = model((X[start : start + batch_size] - mean) / scale)
                probs.append(torch.softmax(logits, dim=1))
                log_odds.append(logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1))
        yield mass, spin, torch.cat(probs).cpu().numpy(), torch.cat(
            log_odds
        ).cpu().numpy()


class ParametricDNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=3,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.35,
    ):
        super(ParametricDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        return self.output_layer(self.network(x))


class ONNXDeployWrapper(nn.Module):
    """Wraps PyTorch model with embedded StandardScaler parameters for standalone ONNX deployment."""

    def __init__(self, trained_model, scaler):
        super(ONNXDeployWrapper, self).__init__()
        self.model = trained_model
        self.register_buffer(
            "mean", torch.tensor(scaler.mean_, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            "scale", torch.tensor(scaler.scale_, dtype=torch.float32).unsqueeze(0)
        )

    def forward(self, x):
        x_scaled = (x - self.mean) / self.scale
        logits = self.model(x_scaled)
        return torch.softmax(logits, dim=1)


class WeightedStandardCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedStandardCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, weights):
        loss = self.ce(logits, targets)
        return (loss * weights).mean()
