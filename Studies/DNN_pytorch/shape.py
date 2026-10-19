import os
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hist import Hist  # Explicitly used for tracking weight variances (fSumw2)

# ==============================================================================
# 1. Configuration & LUT Definition
# ==============================================================================
class_value_LUT = {
    -3: "radion_2L",  # Signal 0
    -2: "bbtautau",  # Signal 1
    -1: "bbWW_1L",  # Signal 2
    0: "bbWW_2L",  # Signal 3 (Primary Signal for Equal-Yield Optimization)
    1: "TT",
    2: "DY",
    3: "ST",
    4: "H",
    5: "VV",
    6: "W",
}

# Color mapping for physics stack plots
COLOR_MAP = {
    "TT": "#54278f",  # Purple
    "DY": "#2b8cbe",  # Blue
    "ST": "#fec44f",  # Orange/Yellow
    "H": "#e31a1c",  # Red
    "VV": "#31a354",  # Green
    "W": "#dd3497",  # Pink
    "radion_2L": "#2ca02c",
    "bbtautau": "#000000",  # Signals overlaid as steps
    "bbWW_1L": "#2ca02c",
    "bbWW_2L": "#d62728",
}

BKG_CLASSES_TO_CHECK = [
    1,
    2,
    3,
    4,
]  # TT, DY, ST, H must strictly be non-zero in all mass bins
CATEGORIES = ["res2b", "recovery", "boosted"]

# Constraints for pDNN splits
MIN_BKG_PER_DNN = 5
N_DNN_CATEGORIES = 4

# High-resolution physical mass range configuration
FINE_MASS_EDGES = np.linspace(250, 2000, 351)

# --- REBINNING MODE CONFIGURATION ---
N_BINS = 10  # Target number of equal signal yield physical mass bins
MIN_BKG_PER_MASS_BIN = (
    0.1  # Minimum background yield required in each variable mass bin
)
REBIN_ON_SUMMED_BKG = False  # True: Rebin on total sum of backgrounds. False: Rebin on individual backgrounds.

# Directories
INPUT_DIR = "looseBTag_multiclass_Radion_pDNN_v7"
OUTPUT_DIR = os.path.join(INPUT_DIR, "combined_validation")
PLOT_DIR = os.path.join(INPUT_DIR, "pdnn_mass_stackplots")


# ==============================================================================
# Helper Function: DP Optimizer for Categorizing pDNN Splits
# ==============================================================================
def get_dnn_edges_dp(df_cat, dnn_col, initial_edges, n_target_bins=3, min_bkg=10.0):
    """Splits the pDNN logit space into n_target_bins maximizing S/sqrt(B) statistical power."""
    print(f"Starting col {dnn_col}")
    mass_suffix = dnn_col.replace("DNN_", "")
    target_mass = float(mass_suffix.replace("M", ""))

    # Filter primary signal events where truth physical mass matches hypothesis target mass
    df_sig = df_cat[(df_cat["class_value"] == 0) & (df_cat["X_mass"] == target_mass)]
    df_all_bkg = df_cat[df_cat["class_value"] > 0]

    if len(df_all_bkg) > 0:
        bkg_scores = df_all_bkg[dnn_col].values
        bkg_weights = df_all_bkg["weight_Central"].values
        total_bkg_counts, _ = np.histogram(
            bkg_scores, bins=initial_edges, weights=bkg_weights
        )
    else:
        total_bkg_counts = np.zeros(len(initial_edges) - 1)

    if len(df_sig) > 0:
        sig_scores = df_sig[dnn_col].values
        sig_weights = df_sig["weight_Central"].values
        sig_counts, _ = np.histogram(
            sig_scores, bins=initial_edges, weights=sig_weights
        )
    else:
        sig_counts = np.zeros(len(initial_edges) - 1)

    M = len(initial_edges) - 1
    cum_sig = np.concatenate(([0.0], np.cumsum(sig_counts)))
    cum_bkg = np.concatenate(([0.0], np.cumsum(total_bkg_counts)))

    memo_valid = {}

    def evaluate_bin(i, j):
        if (i, j) in memo_valid:
            return memo_valid[(i, j)]
        b_val = cum_bkg[j + 1] - cum_bkg[i]
        if b_val < min_bkg:
            res = (False, -1e9)
            memo_valid[(i, j)] = res
            return res
        s_val = cum_sig[j + 1] - cum_sig[i]
        sig_sq = (s_val**2) / b_val if s_val > 0.0 else 0.0
        res = (True, sig_sq)
        memo_valid[(i, j)] = res
        return res

    for t_bins in range(n_target_bins, 0, -1):
        dp = np.full((t_bins + 1, M), -1e9)
        parent = np.full((t_bins + 1, M), -1, dtype=int)

        for k in range(M):
            is_val, sig_sq = evaluate_bin(0, k)
            if is_val:
                dp[1, k] = sig_sq
                parent[1, k] = 0

        for t in range(2, t_bins + 1):
            for k in range(t - 1, M):
                best_val = -1e9
                best_j = -1
                for j in range(t - 1, k + 1):
                    is_val, sig_sq = evaluate_bin(j, k)
                    if is_val:
                        val = dp[t - 1, j - 1] + sig_sq
                        if val > best_val:
                            best_val = val
                            best_j = j
                if best_val > -1e9:
                    dp[t, k] = best_val
                    parent[t, k] = best_j

        if dp[t_bins, M - 1] >= 0.0:
            edges_indices = [M]
            curr_k = M - 1
            for t in range(t_bins, 0, -1):
                curr_j = parent[t, curr_k]
                edges_indices.append(curr_j)
                curr_k = curr_j - 1
            edges_indices.reverse()
            return initial_edges[edges_indices]

    return initial_edges


# ==============================================================================
# 2. Helper Function: DP Equal-Signal-Yield Mass Rebinning
# ==============================================================================
def get_mass_rebinned_edges_dp(
    df_sub, target_mass, initial_edges, n_bins=10, min_bkg=3.0, rebin_on_summed=True
):
    """
    Finds a variable-width mass binning scheme minimizing the variance of the
    primary signal (bbWW_2L) yield across bins. Confined by total or individual background thresholds.
    """
    df_sig = df_sub[(df_sub["class_value"] == 0) & (df_sub["X_mass"] == target_mass)]
    df_all_bkg = df_sub[df_sub["class_value"] > 0]

    if len(df_all_bkg) > 0:
        bkg_masses = df_all_bkg["DeepHME_mass"].values
        bkg_weights = df_all_bkg["weight_Central"].values
        total_bkg, _ = np.histogram(bkg_masses, bins=initial_edges, weights=bkg_weights)
    else:
        total_bkg = np.zeros(len(initial_edges) - 1)

    if len(df_sig) > 0:
        sig_masses = df_sig["DeepHME_mass"].values
        sig_weights = df_sig["weight_Central"].values
        sig_counts, _ = np.histogram(
            sig_masses, bins=initial_edges, weights=sig_weights
        )
    else:
        sig_counts = np.zeros(len(initial_edges) - 1)

    proc_histograms = {}
    active_positivity_classes = []
    for class_val in BKG_CLASSES_TO_CHECK:
        df_proc = df_sub[df_sub["class_value"] == class_val]
        if len(df_proc) > 0:
            counts, _ = np.histogram(
                df_proc["DeepHME_mass"].values,
                bins=initial_edges,
                weights=df_proc["weight_Central"].values,
            )
            if np.sum(counts) > 0:
                proc_histograms[class_val] = counts
                active_positivity_classes.append(class_val)

    M = len(initial_edges) - 1
    cum_sig = np.concatenate(([0.0], np.cumsum(sig_counts)))
    cum_bkg = np.concatenate(([0.0], np.cumsum(total_bkg)))
    cum_proc = {
        c: np.concatenate(([0.0], np.cumsum(proc_histograms[c])))
        for c in active_positivity_classes
    }

    s_total = cum_sig[-1]

    memo_valid = {}

    def evaluate_bin(i, j, target_yield):
        if (i, j) in memo_valid:
            return memo_valid[(i, j)]

        # 1. Evaluate total background threshold constraint
        b_val = cum_bkg[j + 1] - cum_bkg[i]
        if rebin_on_summed:
            if b_val < min_bkg:
                res = (False, 1e18)
                memo_valid[(i, j)] = res
                return res

        # 2. Evaluate individual background constraint thresholds
        for c in active_positivity_classes:
            indiv_bkg_yield = cum_proc[c][j + 1] - cum_proc[c][i]
            if rebin_on_summed:
                # Basic positivity test to make sure individual templates aren't completely empty
                if indiv_bkg_yield <= 0.0:
                    res = (False, 1e18)
                    memo_valid[(i, j)] = res
                    return res
            else:
                # Stricter: Every active individual background must cross the min_bkg limit independently
                if indiv_bkg_yield < min_bkg:
                    res = (False, 1e18)
                    memo_valid[(i, j)] = res
                    return res

        s_val = cum_sig[j + 1] - cum_sig[i]
        penalty = (s_val - target_yield) ** 2
        res = (True, penalty)
        memo_valid[(i, j)] = res
        return res

    for t_bins in range(n_bins, 0, -1):
        s_target = s_total / t_bins if t_bins > 0 else 0.0
        memo_valid.clear()

        dp = np.full((t_bins + 1, M), 1e18)
        parent = np.full((t_bins + 1, M), -1, dtype=int)

        for k in range(M):
            is_val, penalty = evaluate_bin(0, k, s_target)
            if is_val:
                dp[1, k] = penalty
                parent[1, k] = 0

        for t in range(2, t_bins + 1):
            for k in range(t - 1, M):
                best_val = 1e18
                best_j = -1
                for j in range(t - 1, k + 1):
                    is_val, penalty = evaluate_bin(j, k, s_target)
                    if is_val:
                        val = dp[t - 1, j - 1] + penalty
                        if val < best_val:
                            best_val = val
                            best_j = j
                if best_val < 1e17:
                    dp[t, k] = best_val
                    parent[t, k] = best_j

        if dp[t_bins, M - 1] < 1e17:
            edges_indices = [M]
            curr_k = M - 1
            for t in range(t_bins, 0, -1):
                curr_j = parent[t, curr_k]
                edges_indices.append(curr_j)
                curr_k = curr_j - 1
            edges_indices.reverse()
            return initial_edges[edges_indices]

    print(
        f"[WARNING] Could not partition DeepHME_mass with constraints for m{target_mass}. Reverting to single bin."
    )
    failed_edges = np.array([initial_edges[0], initial_edges[-1]])
    return failed_edges


# ==============================================================================
# 3. Dataset Merging (Consolidation over Parity Folders)
# ==============================================================================
dfs_to_merge = []

print("=" * 80)
print("1. MERGING PARITY VALIDATION DATASETS")
print("=" * 80)

for parity in range(4):
    target_dir = os.path.join(INPUT_DIR, f"nParity{parity}_validation")
    input_filepath = os.path.join(target_dir, "validation_applied.root")

    if not os.path.exists(input_filepath):
        print(f"[WARNING] Skipping missing parity fold: '{input_filepath}'")
        continue

    print(f"--> Reading validation tree from '{input_filepath}'...")
    with uproot.open(input_filepath) as f_in:
        tree = f_in["AppliedTree"]
        all_keys = tree.keys()
        dnn_keys = [
            k
            for k in all_keys
            if k.startswith("DNN_M")
            and not k.endswith(("_sig", "_tt", "_other", "_signal"))
        ]
        load_keys = [
            "class_value",
            "weight_Central",
            "res2b",
            "recovery",
            "boosted",
            "X_mass",
            "DeepHME_mass",
        ] + dnn_keys
        df_part = tree.arrays(load_keys, library="pd")
        dfs_to_merge.append(df_part)

if not dfs_to_merge:
    raise RuntimeError(
        "No input validation files found! Please check your parity folder paths."
    )

df = pd.concat(dfs_to_merge, ignore_index=True)
print(f"--> Merging complete. Combined dataset has {len(df):,} total events.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Fine logit limits matching model output ranges (-15 to 15)
fine_dnn_edges = np.linspace(-15, 15, 301)


# ==============================================================================
# Helper Function: Safely Write a Weighted Histogram to ROOT File via Hist
# ==============================================================================
def write_weighted_hist(f_out, name, values, weights, edges, is_signal=False):
    """
    Computes both the sum of weights (values) and sum of weights squared (variances),
    creating a proper Hist object that uproot writes to ROOT with native fSumw2 tracked.
    Also injects a tiny non-zero background baseline if a signal template is empty.
    """
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    variances, _ = np.histogram(values, bins=edges, weights=weights**2)

    # Null norm protection for empty templates (Combine text2workspace safe-guard)
    if is_signal and np.sum(counts) <= 0.0:
        counts = np.full(len(edges) - 1, 1e-9)
        variances = np.full(len(edges) - 1, 1e-18)

    # Initialize a boost-histogram styled Hist container
    h = Hist.new.Variable(edges, name=name).Weight()
    h.view(flow=False).value = counts
    h.view(flow=False).variance = variances

    # Write to root using native object parsing
    f_out[name] = h
    return counts


# ==============================================================================
# 4. Execution Loop Over Categories (Using Merged Data)
# ==============================================================================
print("\n" + "=" * 80)
print(
    f"2. COMPUTING OPTIMAL pDNN SPLITS AND VARIABLE MASS TEMPLATES (Summed Rebin Mode: {REBIN_ON_SUMMED_BKG})"
)
print("=" * 80)


for cat in CATEGORIES:
    output_filename = os.path.join(OUTPUT_DIR, f"shape_{cat}.root")
    print(f"\nCategory: '{cat}' -> Writing to '{output_filename}'...")

    df_cat = df[df[cat] == 1]

    dnn_binning_schemes = {}
    for dnn_col in dnn_keys:
        dnn_binning_schemes[dnn_col] = get_dnn_edges_dp(
            df_cat,
            dnn_col,
            fine_dnn_edges,
            n_target_bins=N_DNN_CATEGORIES,
            min_bkg=MIN_BKG_PER_DNN,
        )

    with uproot.recreate(output_filename) as f_out:
        plot_data_dict = {
            dnn_col: {
                dnn_bin: {"bkg": {}, "sig": {}, "edges": None}
                for dnn_bin in range(N_DNN_CATEGORIES)
            }
            for dnn_col in dnn_keys
        }

        # ----------------------------------------------------------------------
        # A. Fill templates per pDNN split with custom variable-width Mass edges
        # ----------------------------------------------------------------------
        for dnn_col in dnn_keys:
            print(f"Starting col {dnn_col}")
            mass_suffix = dnn_col.replace("DNN_", "")
            target_mass = float(mass_suffix.replace("M", ""))
            dnn_edges = dnn_binning_schemes[dnn_col]

            for dnn_bin_idx in range(len(dnn_edges) - 1):
                dnn_low = dnn_edges[dnn_bin_idx]
                dnn_high = dnn_edges[dnn_bin_idx + 1]

                # Filter category events to specific pDNN channels
                df_sub = df_cat[
                    (df_cat[dnn_col] >= dnn_low) & (df_cat[dnn_col] < dnn_high)
                ]

                # Optimize physical mass boundaries to match equal signal yield per bin
                optimized_mass_edges = get_mass_rebinned_edges_dp(
                    df_sub,
                    target_mass,
                    FINE_MASS_EDGES,
                    n_bins=N_BINS,
                    min_bkg=MIN_BKG_PER_MASS_BIN,
                    rebin_on_summed=REBIN_ON_SUMMED_BKG,
                )

                # Save optimized edges to plot metadata
                plot_data_dict[dnn_col][dnn_bin_idx]["edges"] = optimized_mass_edges

                # Fill data_obs via robust tracking function
                df_bkg_only = df_sub[df_sub["class_value"] > 0]
                hist_name_obs = f"data_obs_{mass_suffix}_dnnbin{dnn_bin_idx}"

                _ = write_weighted_hist(
                    f_out,
                    hist_name_obs,
                    values=df_bkg_only["DeepHME_mass"].values,
                    weights=df_bkg_only["weight_Central"].values,
                    edges=optimized_mass_edges,
                    is_signal=False,
                )

                # Initialize accumulator for the total background template sum tracking
                total_bkg_counts = np.zeros(len(optimized_mass_edges) - 1)
                total_bkg_variances = np.zeros(len(optimized_mass_edges) - 1)

                # Fill individual processes
                for class_val, process_name in class_value_LUT.items():
                    df_proc = df_sub[df_sub["class_value"] == class_val]
                    hist_name_proc = f"{process_name}_{mass_suffix}_dnnbin{dnn_bin_idx}"

                    if class_val <= 0:
                        df_step = df_proc[df_proc["X_mass"] == target_mass]
                        is_sig_proc = True
                    else:
                        df_step = df_proc
                        is_sig_proc = False

                    # Write templates correctly tracking errors and handling empty slots gracefully
                    counts = write_weighted_hist(
                        f_out,
                        hist_name_proc,
                        values=df_step["DeepHME_mass"].values,
                        weights=df_step["weight_Central"].values,
                        edges=optimized_mass_edges,
                        is_signal=is_sig_proc,
                    )

                    if class_val > 0:
                        plot_data_dict[dnn_col][dnn_bin_idx]["bkg"][
                            process_name
                        ] = counts
                        total_bkg_counts += counts
                        # Track separate variance calculations for the sum plot loop baseline
                        bkg_vars, _ = np.histogram(
                            df_step["DeepHME_mass"].values,
                            bins=optimized_mass_edges,
                            weights=df_step["weight_Central"].values ** 2,
                        )
                        total_bkg_variances += bkg_vars
                    else:
                        plot_data_dict[dnn_col][dnn_bin_idx]["sig"][
                            process_name
                        ] = counts

                # Write the total accumulated background template using the native Hist framework
                hist_name_bkg_total = f"background_{mass_suffix}_dnnbin{dnn_bin_idx}"
                h_bkg = Hist.new.Variable(
                    optimized_mass_edges, name=hist_name_bkg_total
                ).Weight()
                h_bkg.view(flow=False).value = total_bkg_counts
                h_bkg.view(flow=False).variance = total_bkg_variances
                f_out[hist_name_bkg_total] = h_bkg

        print(
            f"      - Filled all variable-width physical mass templates (N_BINS goal: {N_BINS})."
        )

        # ----------------------------------------------------------------------
        # B. Generate Stack Plots Normalized by Bin Width
        # ----------------------------------------------------------------------
        print(f"      - Generating stack plots for category '{cat}'...")

        for dnn_col in dnn_keys:
            mass_suffix = dnn_col.replace("DNN_", "")

            fig, axes = plt.subplots(
                1, N_DNN_CATEGORIES, figsize=(6 * N_DNN_CATEGORIES, 5), sharey=False
            )
            if N_DNN_CATEGORIES == 1:
                axes = [axes]

            dnn_edges = dnn_binning_schemes[dnn_col]

            for dnn_bin_idx in range(N_DNN_CATEGORIES):
                ax = axes[dnn_bin_idx]
                edges = plot_data_dict[dnn_col][dnn_bin_idx]["edges"]
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                bin_widths = np.diff(edges)

                bkg_yields = plot_data_dict[dnn_col][dnn_bin_idx]["bkg"]
                sorted_bkg_names = sorted(
                    bkg_yields.keys(), key=lambda k: np.sum(bkg_yields[k])
                )

                stack_y = []
                stack_colors = []
                stack_labels = []

                for bproc in sorted_bkg_names:
                    if np.sum(bkg_yields[bproc]) > 0:
                        normalized_counts = bkg_yields[bproc] / bin_widths
                        stack_y.append(normalized_counts)
                        stack_colors.append(COLOR_MAP.get(bproc, "#969696"))
                        stack_labels.append(f"{bproc}: {np.sum(bkg_yields[bproc]):.2f}")

                if len(stack_y) > 0:
                    ax.hist(
                        [bin_centers] * len(stack_y),
                        bins=edges,
                        weights=stack_y,
                        stacked=True,
                        color=stack_colors,
                        label=stack_labels,
                        histtype="stepfilled",
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.85,
                    )

                sig_yields = plot_data_dict[dnn_col][dnn_bin_idx]["sig"]
                for sproc, scounts in sig_yields.items():
                    sig_sum = np.sum(scounts)
                    if sig_sum > 0:
                        normalized_sig = scounts / bin_widths
                        bkg_tot = (
                            sum(np.sum(by) for by in stack_y)
                            if len(stack_y) > 0
                            else 1.0
                        )
                        scale_factor = 1.0
                        if sig_sum < (bkg_tot * 0.05):
                            scale_factor = 10.0
                            if sig_sum < (bkg_tot * 0.005):
                                scale_factor = 100.0

                        ax.hist(
                            bin_centers,
                            bins=edges,
                            weights=normalized_sig * scale_factor,
                            histtype="step",
                            color=COLOR_MAP.get(sproc, "red"),
                            linewidth=2.0,
                            label=(
                                f"{sproc}: {np.sum(scounts):.2f}"
                                if scale_factor == 1.0
                                else f"{sproc} (x{int(scale_factor)}): {np.sum(scounts):.2f}"
                            ),
                        )

                dnn_bin_labels = [
                    "Low pDNN Bin",
                    "Medium pDNN Bin",
                    "High pDNN Bin",
                    "Very High pDNN Bin",
                ]
                title_label = (
                    dnn_bin_labels[dnn_bin_idx]
                    if dnn_bin_idx < len(dnn_bin_labels)
                    else f"pDNN Bin {dnn_bin_idx}"
                )
                ax.set_title(
                    f"{title_label}\n({dnn_edges[dnn_bin_idx]:.2f} < pDNN < {dnn_edges[dnn_bin_idx+1]:.2f})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel(r"DeepHME_mass [GeV]", fontsize=10)
                ax.set_ylabel("Weighted Events / GeV", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.set_xlim(edges[0], edges[-1])

                if dnn_bin_idx == N_DNN_CATEGORIES - 1:
                    ax.set_yscale("log")

                ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

            plt.suptitle(
                f"Category: {cat} | Mass Hypothesis: {mass_suffix}",
                fontsize=14,
                y=1.02,
                fontweight="bold",
            )
            plt.tight_layout()

            plot_savepath = os.path.join(PLOT_DIR, f"stack_{cat}_{mass_suffix}.png")
            plt.savefig(plot_savepath, dpi=150, bbox_inches="tight")
            plt.close()

print(
    f"\n--> Done! Variable physical mass templates optimized for equal signal yield (N_BINS={N_BINS}) are saved inside '{OUTPUT_DIR}/'."
)
