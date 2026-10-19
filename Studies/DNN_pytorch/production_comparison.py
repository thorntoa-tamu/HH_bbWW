#!/usr/bin/env python3
"""
Puts the new pDNN in the context of the analysis: Run 2 DL expected limits, the latest production
DL expected limits (deployed model, full analysis with systematics), and a projection of the
production limits with the new model.

The quick limits of limit_pdnn.py are stat-only, so they are not compared with the full-analysis
limits directly. Instead the deployed model is run through the same quick chain (score_deployed.py),
and the production limit is scaled by the ratio new / deployed of the quick limits:
    projected(m) = production(m) * quick_new(m) / quick_deployed(m)
All limits are expected 95% CL limits on sigma(pp -> X -> HH) in pb, spin 0 (Radion).

Usage:
    python3 production_comparison.py --deployed-limits <json> --new-limits <json> \
        --production-csv <limits_table.csv> --run2-json <json> --output-dir <dir> [--category combined]
"""

import os
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def quick(path, category, spin="spin0"):
    r = json.load(open(path))[spin][category]
    return {int(m): v for m, v in r.items() if v and v.get("median")}


def main():
    p = argparse.ArgumentParser(
        description="Compare the new pDNN with production and Run 2"
    )
    p.add_argument(
        "--deployed-limits",
        required=True,
        help="limit_pdnn.py json of the deployed model",
    )
    p.add_argument(
        "--new-limits", required=True, help="limit_pdnn.py json of the new model"
    )
    p.add_argument(
        "--production-csv",
        required=True,
        help="table with mX_GeV and DL_* columns (pb)",
    )
    p.add_argument("--run2-json", required=True, help="Run 2 DL expected limits (pb)")
    p.add_argument("--category", default="combined")
    p.add_argument("--label-new", default="new pDNN")
    p.add_argument("--label-deployed", default="production pDNN")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    dep = quick(a.deployed_limits, a.category)
    new = quick(a.new_limits, a.category)
    prod = {}
    for row in csv.DictReader(open(a.production_csv)):
        if row.get("DL_median"):
            prod[int(row["mX_GeV"])] = {
                k: float(row[f"DL_{k}"])
                for k in ("median", "m1sigma", "p1sigma", "m2sigma", "p2sigma")
            }
    run2 = {
        int(float(m)): v for m, v in json.load(open(a.run2_json))[0]["limits"].items()
    }

    rows = []
    for m in sorted(prod):
        if m in dep and m in new:
            ratio = new[m]["median"] / dep[m]["median"]
            rows.append(
                {
                    "mX_GeV": m,
                    "run2_DL": run2.get(m, ""),
                    "production_DL": prod[m]["median"],
                    "production_DL_m1": prod[m]["m1sigma"],
                    "production_DL_p1": prod[m]["p1sigma"],
                    "quick_deployed": dep[m]["median"],
                    "quick_new": new[m]["median"],
                    "ratio_new_over_deployed": ratio,
                    "projected_DL_new": prod[m]["median"] * ratio,
                    "projected_over_run2": (
                        (prod[m]["median"] * ratio / run2[m]) if m in run2 else ""
                    ),
                    "production_over_run2": (
                        (prod[m]["median"] / run2[m]) if m in run2 else ""
                    ),
                }
            )
    out_csv = os.path.join(a.output_dir, f"production_comparison_{a.category}.csv")
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    m = np.array([r["mX_GeV"] for r in rows])
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    ax.fill_between(
        m,
        [r["production_DL_m1"] for r in rows],
        [r["production_DL_p1"] for r in rows],
        color="gold",
        alpha=0.6,
        label=r"production $\pm 1\sigma$",
    )
    ax.plot(
        m,
        [r["production_DL"] for r in rows],
        "k--",
        label=f"production DL ({a.label_deployed}, full analysis)",
    )
    ax.plot(
        m,
        [r["projected_DL_new"] for r in rows],
        "o-",
        color="tab:red",
        label=f"projected with {a.label_new}",
    )
    r2 = sorted(run2.items())
    ax.plot(
        [x for x, _ in r2],
        [y for _, y in r2],
        "s-",
        color="#832db6",
        label=r"Run 2 DL expected, 138 fb$^{-1}$",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"expected 95% CL limit on $\sigma(pp\to X\to HH)$ [pb]")
    ax.set_title("Spin 0, dilepton: 62.3 fb$^{-1}$ (13.6 TeV)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    rax.plot(
        m,
        [r["ratio_new_over_deployed"] for r in rows],
        "o-",
        color="tab:red",
        label="new / production model (quick chain)",
    )
    pm = [r["mX_GeV"] for r in rows if r["projected_over_run2"] != ""]
    rax.plot(
        pm,
        [r["projected_over_run2"] for r in rows if r["projected_over_run2"] != ""],
        "s-",
        color="#832db6",
        label="projected / Run 2",
    )
    rax.plot(
        pm,
        [r["production_over_run2"] for r in rows if r["production_over_run2"] != ""],
        "s--",
        color="gray",
        label="production / Run 2",
    )
    rax.axhline(1, color="black", lw=0.8)
    rax.set_xlabel("$m_X$ [GeV]")
    rax.set_ylabel("ratio")
    rax.grid(True, alpha=0.3)
    rax.legend(fontsize=7)
    plt.tight_layout()
    out_png = os.path.join(a.output_dir, f"production_comparison_{a.category}.png")
    plt.savefig(out_png, dpi=150)
    print(f"--> Wrote {out_csv} and {out_png}")
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
