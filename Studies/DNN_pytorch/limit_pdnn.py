#!/usr/bin/env python3
"""
Quick expected limits from a trained pDNN, to compare models.

Reads validation_applied.root of every parity cycle (each event is in exactly one validation
fold, so together they hold every event once, at its physical weight_Central), builds per-mass
templates of the pDNN signal logit in the res2b / recovery / boosted categories, and runs
stat-only expected limits (combine -M AsymptoticLimits -t -1) per category and combined.

The signal of a spin hypothesis is every class_value of that spin among signal_classes
(e.g. spin 0: Radion 2L + bbtautau) at the hypothesis mass; backgrounds are grouped into
TT / DY / Other. Limits are on the signal strength r of the dataset's signal normalization,
so they compare models trained and evaluated on the same dataset.

Usage:
    python3 limit_pdnn.py --config model_configs/v2610_pDNN_v1.yaml
    python3 limit_pdnn.py --compare <limits.json> <limits.json> ... --labels A B ...
"""

import os
import json
import argparse
import subprocess
import yaml
import uproot
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import ROOT

from src.model_helper import eval_parity_expr, get_class_setup, is_signal

ROOT.gROOT.SetBatch(True)

CATEGORIES = ["res2b", "recovery", "boosted"]
BACKGROUNDS = {"TT": [1], "DY": [2], "Other": "rest"}


def validation_paths(cfg):
    paths = []
    for i_fold in range(cfg.get("nParity", 4)):
        train_parity = eval_parity_expr(cfg["train_parity"]["index"], i_fold)
        paths.append(
            os.path.join(
                cfg["output_folder"],
                f"nParity{train_parity}_validation",
                "validation_applied.root",
            )
        )
    return paths


def load_validation(cfg, masses, spins, extra=()):
    """
    Reads only the columns the limits need: the bookkeeping branches and one signal logit per
    hypothesis (DNN_M{m}, or DNN_M{m}_S{spin} for spin-parametric models). The validation
    trees also hold the class probabilities of every hypothesis, which are not needed here.
    Returns the arrays and whether the scores are per spin.
    """
    paths = validation_paths(cfg)
    tree_name = cfg.get("tree_name", "AppliedTree")
    with uproot.open(paths[0]) as f:
        available = set(f[tree_name].keys())
    spin_is_input = any(k.endswith("_S0") or k.endswith("_S2") for k in available)
    scores = [
        f"DNN_M{m}_S{s}" if spin_is_input else f"DNN_M{m}"
        for m in masses
        for s in spins
    ]
    base = ["class_value", "weight_Central", "X_mass", "boosted", "res2b", "recovery"]
    base += list(extra)
    branches = sorted(set(base) | {b for b in scores if b in available})
    arrays = []
    for path in paths:
        with uproot.open(path) as f:
            arrays.append(f[tree_name].arrays(branches, library="np"))
    return {k: np.concatenate([a[k] for a in arrays]) for k in branches}, spin_is_input


def category_masks(data):
    boosted = data["boosted"] == 1
    return {
        "boosted": boosted,
        "res2b": ~boosted & (data["res2b"] == 1),
        "recovery": ~boosted & (data["res2b"] != 1) & (data["recovery"] == 1),
    }


def compute_edges(score_sig, w_sig, score_bkg, w_bkg, n_bins, min_bkg, min_bkg_events):
    """
    Candidate edges at the equal-yield quantiles of the signal and of the background, then
    merged from the high-score end until each bin has at least min_bkg total background yield
    and min_bkg_events background MC events. The background quantiles keep an edge between a
    signal that sits entirely at high score and the background bulk.
    """
    lo = min(score_sig.min(), score_bkg.min()) - 1e-3
    hi = max(score_sig.max(), score_bkg.max()) + 1e-3

    def quantiles(values, weights, fractions):
        order = np.argsort(values)
        cum = np.cumsum(weights[order]) / weights.sum()
        idx = np.clip(np.searchsorted(cum, fractions), 0, len(values) - 1)
        return values[order][idx]

    uniform = np.arange(1, n_bins) / n_bins
    # Background tail: fractions 1 - 10^-k in half-decades, so the top bin can stay small
    tail = 1 - np.logspace(-1, -6, 11)
    edges = np.unique(
        np.r_[
            lo,
            quantiles(score_sig, w_sig, uniform),
            quantiles(score_bkg, w_bkg, np.r_[uniform, tail]),
            hi,
        ]
    )

    merged = [edges[-1]]
    i = len(edges) - 1
    while i > 0:
        j = i - 1
        while j > 0:
            in_bin = (score_bkg >= edges[j]) & (score_bkg < edges[i])
            if w_bkg[in_bin].sum() >= min_bkg and in_bin.sum() >= min_bkg_events:
                break
            j -= 1
        merged.append(edges[j])
        i = j
    return np.array(sorted(set(merged)))


def merge_left_to_right(values_bkg, w_bkg, edges, min_bkg, min_bkg_events):
    """Merges neighbouring bins from low to high until each has the minimum background."""
    merged = [edges[0]]
    i = 0
    while i < len(edges) - 1:
        j = i + 1
        while j < len(edges) - 1:
            in_bin = (values_bkg >= edges[i]) & (values_bkg < edges[j])
            if w_bkg[in_bin].sum() >= min_bkg and in_bin.sum() >= min_bkg_events:
                break
            j += 1
        merged.append(edges[j])
        i = j
    # A last bin short of background is merged into its neighbour
    if len(merged) > 2:
        last = (values_bkg >= merged[-2]) & (values_bkg < merged[-1])
        if w_bkg[last].sum() < min_bkg or last.sum() < min_bkg_events:
            del merged[-2]
    return np.array(merged)


def binning_2d(score_s, hme_s, w_s, score_b, hme_b, w_b, args):
    """
    DNN score x DeepHME_mass binning: the DNN slices are exactly the 1D binning, and each slice is
    split in HME at the signal quantiles, merging neighbouring HME bins until each has the
    minimum background. The 2D binning is then a refinement of the 1D one, so the comparison
    shows what DeepHME_mass adds on top of the same DNN binning.
    """
    dnn_edges = compute_edges(
        score_s, w_s, score_b, w_b, args.n_bins, args.min_bkg, args.min_bkg_events
    )
    hme_lo = min(hme_s.min(), hme_b.min()) - 1e-3
    hme_hi = max(hme_s.max(), hme_b.max()) + 1e-3
    hme_edges = []
    for k in range(len(dnn_edges) - 1):
        in_s = (score_s >= dnn_edges[k]) & (score_s < dnn_edges[k + 1])
        in_b = (score_b >= dnn_edges[k]) & (score_b < dnn_edges[k + 1])
        cand = [hme_lo, hme_hi]
        if w_s[in_s].sum() > 0:
            order = np.argsort(hme_s[in_s])
            cum = np.cumsum(w_s[in_s][order]) / w_s[in_s].sum()
            q = np.arange(1, args.n_hme_bins) / args.n_hme_bins
            idx = np.clip(np.searchsorted(cum, q), 0, in_s.sum() - 1)
            cand += list(hme_s[in_s][order][idx])
        hme_edges.append(
            merge_left_to_right(
                hme_b[in_b],
                w_b[in_b],
                np.unique(cand),
                args.min_bkg,
                args.min_bkg_events,
            )
        )
    return dnn_edges, hme_edges


def unroll(score, hme, binning):
    """Position of each event in the unrolled 2D histogram (bin k at [k, k+1))."""
    dnn_edges, hme_edges = binning
    pos = np.full(len(score), -1.0)
    offset = 0
    for k, edges in enumerate(hme_edges):
        in_slice = (score >= dnn_edges[k]) & (score < dnn_edges[k + 1])
        idx = np.clip(np.digitize(hme[in_slice], edges) - 1, 0, len(edges) - 2)
        pos[in_slice] = offset + idx + 0.5
        offset += len(edges) - 1
    return pos, offset


def make_hist(name, values, weights, edges):
    h = ROOT.TH1D(name, name, len(edges) - 1, np.asarray(edges, dtype=np.float64))
    h.Sumw2()
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    sumw2, _ = np.histogram(values, bins=edges, weights=weights**2)
    for b in range(len(counts)):
        # Templates must be positive for combine
        h.SetBinContent(b + 1, max(counts[b], 1e-6))
        h.SetBinError(b + 1, np.sqrt(sumw2[b]) if counts[b] > 0 else 1e-6)
    return h


def write_card(path, category, shapes_file, processes):
    names = list(processes)
    lines = [
        "imax 1",
        f"jmax {len(names) - 1}",
        "kmax *",
        "-" * 40,
        f"shapes * {category} {shapes_file} $PROCESS",
        "-" * 40,
        f"bin {category}",
        "observation -1",
        "-" * 40,
        "bin " + " ".join([category] * len(names)),
        "process " + " ".join(names),
        "process " + " ".join(str(i) for i in range(len(names))),
        "rate " + " ".join(["-1"] * len(names)),
        "-" * 40,
        f"{category} autoMCStats 10 0 1",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_combine(workdir, card, name, mass):
    cmd = [
        "combine",
        "-M",
        "AsymptoticLimits",
        card,
        "-t",
        "-1",
        "--run",
        "blind",
        # The default tolerance (rAbsAcc 0.0005) quantizes limits of r ~ 0.02 in ~2.5% steps,
        # too coarse to compare models
        "--rAbsAcc",
        "1e-6",
        "--rRelAcc",
        "1e-3",
        "-n",
        f"_{name}",
        "-m",
        str(mass),
    ]
    subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    out = os.path.join(workdir, f"higgsCombine_{name}.AsymptoticLimits.mH{mass}.root")
    if not os.path.exists(out):
        return None
    with uproot.open(out) as f:
        t = f["limit"].arrays(["limit", "quantileExpected"], library="np")
    # quantileExpected is float32 (0.16 -> 0.1599999964): round as Python floats
    q = {
        round(float(k), 3): float(v) for k, v in zip(t["quantileExpected"], t["limit"])
    }
    return {
        "median": q.get(0.5),
        "m1": q.get(0.16),
        "p1": q.get(0.84),
        "m2": q.get(0.025),
        "p2": q.get(0.975),
    }


def w_sum(weight, mask):
    return float(weight[mask].sum())


def limits_for_model(cfg, args):
    class_setup = get_class_setup(cfg)
    signal_classes = class_setup["signal_classes"]
    masses = args.masses or cfg["signal_mass_points"]
    spins = args.spins
    extra = ["DeepHME_mass"] if args.observable == "dnn_hme" else []
    data, spin_is_input = load_validation(cfg, masses, spins, extra)
    cats = category_masks(data)
    class_value = data["class_value"]
    weight = data["weight_Central"].astype(np.float64)

    bkg_masks = {}
    listed = []
    for name, spec in BACKGROUNDS.items():
        if spec != "rest":
            bkg_masks[name] = np.isin(class_value, spec)
            listed += spec
    bkg_masks["Other"] = (class_value > 0) & ~np.isin(class_value, listed)
    any_bkg = class_value > 0

    out_dir = args.output_dir or os.path.join(
        cfg["output_folder"], "limits" if args.observable == "dnn" else "limits_dnn_hme"
    )
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for spin in spins:
        spin_sig = is_signal(class_value, class_setup) & np.isin(
            class_value,
            [int(c) for c, s in class_setup["signal_spin"].items() if s == spin],
        )
        for mass in masses:
            score_name = f"DNN_M{mass}_S{spin}" if spin_is_input else f"DNN_M{mass}"
            if score_name not in data:
                print(f"[WARNING] {score_name} not in the validation trees, skipping")
                continue
            score = data[score_name]
            sig = spin_sig & (data["X_mass"] == mass)
            workdir = os.path.join(out_dir, f"spin{spin}_M{mass}")
            os.makedirs(workdir, exist_ok=True)
            cards = []
            for cat in args.categories:
                cmask = cats[cat]
                s_mask = sig & cmask
                b_mask = any_bkg & cmask
                if w_sum(weight, s_mask) <= 0 or b_mask.sum() < args.min_bkg_events:
                    continue
                if args.observable == "dnn_hme":
                    hme = data["DeepHME_mass"]
                    binning = binning_2d(
                        score[s_mask],
                        hme[s_mask],
                        weight[s_mask],
                        score[b_mask],
                        hme[b_mask],
                        np.clip(weight[b_mask], 0, None),
                        args,
                    )
                    x, n_total = unroll(score, hme, binning)
                    edges = np.arange(n_total + 1, dtype=np.float64)
                else:
                    x = score
                    edges = compute_edges(
                        score[s_mask],
                        weight[s_mask],
                        score[b_mask],
                        np.clip(weight[b_mask], 0, None),
                        args.n_bins,
                        args.min_bkg,
                        args.min_bkg_events,
                    )
                shapes = f"shapes_{cat}.root"
                f = ROOT.TFile(os.path.join(workdir, shapes), "RECREATE")
                hists = [make_hist("Signal", x[s_mask], weight[s_mask], edges)]
                for bname, bmask in bkg_masks.items():
                    m = bmask & cmask
                    hists.append(make_hist(bname, x[m], weight[m], edges))
                # Asimov expected limits ignore data_obs, but combine needs it in the file
                data_obs = hists[1].Clone("data_obs")
                for h in hists[2:]:
                    data_obs.Add(h)
                for h in hists + [data_obs]:
                    h.Write()
                f.Close()
                card = f"card_{cat}.txt"
                write_card(
                    os.path.join(workdir, card),
                    cat,
                    shapes,
                    ["Signal"] + list(bkg_masks),
                )
                cards.append((cat, card))
            if not cards:
                continue
            if len(cards) > 1:
                combined = "card_combined.txt"
                with open(os.path.join(workdir, combined), "w") as fout:
                    subprocess.run(
                        ["combineCards.py"] + [f"{c}={card}" for c, card in cards],
                        cwd=workdir,
                        stdout=fout,
                        check=True,
                    )
                cards.append(("combined", combined))
            for cat, card in cards:
                jobs.append((spin, mass, cat, workdir, card))

    def run(job):
        spin, mass, cat, workdir, card = job
        return job, run_combine(workdir, card, cat, mass)

    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for (spin, mass, cat, _, _), lim in ex.map(run, jobs):
            results.setdefault(f"spin{spin}", {}).setdefault(cat, {})[str(mass)] = lim

    out_json = os.path.join(out_dir, "limits.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"--> Wrote {out_json}")
    return out_json


def plot_limits(json_paths, labels, out_path, category="combined"):
    results = [json.load(open(p)) for p in json_paths]
    spins = sorted({s for r in results for s in r})
    fig, axes = plt.subplots(1, len(spins), figsize=(7 * len(spins), 5), squeeze=False)
    for ax, spin in zip(axes[0], spins):
        for r, label in zip(results, labels):
            lims = r.get(spin, {}).get(category, {})
            pts = sorted(
                (int(m), v) for m, v in lims.items() if v and v["median"] is not None
            )
            if not pts:
                continue
            m = np.array([p[0] for p in pts])
            med = np.array([p[1]["median"] for p in pts])
            line = ax.plot(m, med, marker="o", label=label)[0]
            if len(results) == 1 and all(p[1]["m1"] is not None for p in pts):
                ax.fill_between(
                    m,
                    [p[1]["m1"] for p in pts],
                    [p[1]["p1"] for p in pts],
                    color=line.get_color(),
                    alpha=0.2,
                )
        ax.set_yscale("log")
        ax.set_xlabel("$m_X$ [GeV]")
        ax.set_ylabel("Expected 95% CL limit on r (stat. only)")
        ax.set_title(f"{spin}, {category}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"--> Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick expected limits for pDNN models"
    )
    parser.add_argument(
        "--config", help="Model config whose output_folder holds the trained folds"
    )
    parser.add_argument("--masses", type=int, nargs="+", default=None)
    parser.add_argument(
        "--observable",
        choices=["dnn", "dnn_hme"],
        default="dnn",
        help="dnn: 1D pDNN log-odds; dnn_hme: the 1D binning split further in DeepHME_mass, unrolled",
    )
    parser.add_argument(
        "--n-dnn-slices", type=int, default=3, help="2D: equal-signal DNN slices"
    )
    parser.add_argument(
        "--n-hme-bins",
        type=int,
        default=6,
        help="2D: HME bins per slice before merging",
    )
    parser.add_argument("--spins", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument(
        "--n-bins", type=int, default=10, help="Equal-signal bins before merging"
    )
    parser.add_argument(
        "--min-bkg", type=float, default=1.0, help="Minimum background yield per bin"
    )
    parser.add_argument(
        "--min-bkg-events",
        type=int,
        default=10,
        help="Minimum background MC events per bin",
    )
    parser.add_argument(
        "--jobs", type=int, default=8, help="Parallel combine processes"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--compare", nargs="+", default=None, help="limits.json files to overlay"
    )
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()

    if args.compare:
        labels = args.labels or [
            os.path.basename(os.path.dirname(os.path.dirname(p))) for p in args.compare
        ]
        out = args.output_dir or "."
        for cat in args.categories + ["combined"]:
            plot_limits(
                args.compare,
                labels,
                os.path.join(out, f"limit_comparison_{cat}.png"),
                cat,
            )
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_json = limits_for_model(cfg, args)
    label = os.path.basename(cfg["output_folder"].rstrip("/"))
    for cat in args.categories + ["combined"]:
        plot_limits(
            [out_json],
            [label],
            os.path.join(os.path.dirname(out_json), f"limits_{cat}.png"),
            cat,
        )


if __name__ == "__main__":
    main()
