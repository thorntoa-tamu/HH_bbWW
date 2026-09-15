# DY 2D adaptive-grid fit

Smooth Drell–Yan template in the (DNN, HME) plane for the `res2b` category of
the resonant X→HH→bbWW search, one fit per mass hypothesis (mX = 300, 400, 500,
550, 600, 650, 700, 800, 900, 1000 GeV).  The DY MC is too sparse to be used
bin by bin, so the fine TH2 (150 × 250 bins) is first merged into an adaptive
non-uniform cell grid with no negative cell, a continuous **ExpPolyLogY2D**
density is fitted to the cells with ROOT Minuit2, and the fitted density is
integrated back onto any binning to produce the nominal template and its
shape/normalisation variations.  Only `core.py`, `fit_one.py`, `grid_merge.py`,
`make_hist_from_fit.py`, `plot_fit_slices.py` and this file are published; the
campaign drivers of the development repository are not shipped.

## Pipeline

| Step | Tool | Input → output |
|------|------|----------------|
| 1 | `grid_merge.py` | fine `TH2` → adaptive cell grid (JSON) + inspection PDF |
| 2 | `fit_one.py` | grid JSON → fit JSON (parameters, covariance, GoF, closure); automatic degree × thr scan |
| 3 | `make_hist_from_fit.py` | fit JSON (+ binning) → `TH2` templates: nominal, accepted-unc, eigenmode up/down |
| 4 | `plot_fit_slices.py` | fit JSON + hadd file → DY vs fit HME projections in DNN slices (PDF) |
| — | `core.py` | shared library: TH2 I/O, density models, cell/bin integration, eigenmodes |

## Quick start

All commands run from this directory with the ROOT-enabled Python
(`python3 -u`, so Minuit messages are not buffered).  `$DATA` holds the hadd
files described in *Data layout*.

```bash
DATA=/path/to/hadd_files      # contains hadd_m{MX}_res2b.root

# 1. adaptive grid from the fine DY TH2
python3 -u grid_merge.py --input $DATA/hadd_m650_res2b.root --hist plots_2d/DY_2d \
    --mx 650 --output-json grid_m650.json --output-pdf grid_m650.pdf

# 2. fit (v2 preset; auto degree 1…8 × thr 0…10 %, first PASS wins)
python3 -u fit_one.py --grid grid_m650.json --output fit_m650.json --preset v2

# 3. templates on the binning of the input TH2, with covariance eigenmodes
python3 -u make_hist_from_fit.py --fit fit_m650.json \
    --from-hist $DATA/hadd_m650_res2b.root:plots_2d/DY_2d \
    -o templates_m650.root --shape-variations

# 4. DY vs fit HME slices, 5 slices of equal signal yield
python3 -u plot_fit_slices.py --fit fit_m650.json --data-dir $DATA \
    --auto-dnn-slice-n-const-signal-bins 5 --output slices_m650.pdf
```

`fit_one.py` exits 0 on PASS and 2 on FAIL (the JSON is written in both
cases; configuration errors such as a misaligned tiling window exit 1 without
a JSON).  Fixed configurations: `--degree 4 --thr 0.1` (no scan),
`--degree 4 --auto-thr` (scan thr only).  Without `--preset v2` the legacy
`v1` behaviour is used (see *Presets* and *Status*).  Explicit
`--dnn-min/--dnn-max/--hme-min/--hme-max` override the automatic fit window
(bounding box of the cells with content > 0); with `--norm tiling` the
explicit edges must lie on fine-bin edges and must not cut through any cell
(a cut cell is an error whether its centre lies inside or outside the window,
so every in-window fine bin belongs to an observation).

## Model

```
s(x,y)  = exp( Σ a_ij · xn^i · yn_log^j ),   1 ≤ i+j ≤ degree
pdf     = s / ∬_window s
μ_cell  = N · ∬_cell pdf
```

- `xn` maps DNN linearly to [−1, 1] over the fit window.
- `yn_log = 2·log1p((y − ymin + ε)/ε) / log1p((ymax − ymin + ε)/ε) − 1` maps
  HME to [−1, 1]; `ε = --y-log-eps` (default 1e-3 GeV, stored as `y_eps`).
- Parameters: `N` plus one `a_ij` per monomial (`npar = (d+1)(d+2)/2`).
- Empty cells: all zero-content cells form one meta-observation (`--no-empty-meta`
  gives one term per empty cell).
- Cell integrals: Gauss–Legendre, `--n-quad` nodes per axis (default 4).

**Normalisation.** With `--norm window_gl` (legacy) the window integral is a
single `n_quad × n_quad` Gauss–Legendre rule over the whole window, which
mis-normalises the steep log-HME density; the template yield then depends on
the quadrature order.  With `--norm tiling` the window integral is the sum of
per-fine-bin Gauss–Legendre integrals (`--n-quad-fine` nodes per axis per fine
bin, default 2) over the fine bins of the grid, every cell μ is the sum of its
fine-bin integrals, and therefore `Σ_cells μ = N` exactly.  The fit JSON stores
`fine_binning` and `function.norm = "tiling"`; `core.integrate_model_bins`
reuses the same fine-bin integrals, so the nominal template integrates to
`sum_mu` exactly on the fine binning and to within numerical precision on any
other binning for fits with `clip_hit = false`.  A fit with `clip_hit = true`
exceeded the +50 exponent clip at a quadrature node and its density is not
reliable off the fine grid (`make_hist_from_fit.py` warns).  A legacy fit
JSON without `stat`/`norm` fields is treated as `chi2_neyman`/`window_gl`
and labelled accordingly.

## Fit statistic and goodness-of-fit

**Neyman χ² (`--stat chi2_neyman`).** `χ² = Σ (c − μ)² / σ²` with
`σ² = σ_stat² + (thr·|c|)²`, `σ_stat` the observed cell error (Garwood upper
error for `c = 0`).  `p = TMath::Prob(χ², n_obs − npar)`.  With thousands of
single-MC-event cells the observed error is not a measure of the expected
fluctuation, and the statistic is blind to the overall yield (see *Status*).

**Scaled Poisson likelihood (`--stat poisson_eff`).** Bohm–Zech −2lnL for
weighted MC: per observation `w' = w_eff + thr²·c`, `n = c/w'`, `λ = μ/w'`,
`−2lnL = 2 Σ [λ − n + n ln(n/λ)]` (empty meta-observation: `n = 0`).  The
effective weight of a positive cell is `w = e²/c` if its `n_eff = c²/e²` is at
least `--weight-neff-min` (10); otherwise the weight of its `--super-cell NX NY`
(10 × 10 fine bins) if that has `n_eff ≥ 10`; otherwise the global
`w_glob = Σe²/Σc` (`--weight-map global` skips the local step; empty
observations use `w_glob`).  `weight_map_stats` records the class counts.

**Accepted uncertainty `thr`.** The relative uncertainty tolerated on the
template, scanned from 0 to `--thr` (default 0.10) in `--thr-step` (0.01)
steps for each degree; the first PASS fixes `thr_min`, which downstream code
turns into the accepted-unc systematic.

**Acceptance GoF for `poisson_eff`: super-cell Pearson χ².** The calibrated
deviance is under-dispersed for heterogeneous weights (p ≈ 1 for every
degree), so acceptance uses a Pearson χ² on super-cells built from the
observations before fitting (fit-independent, `build_gof_supercells`):
positive cells are assigned by their centre to DNN bands of `--gof-band` fine
bins (10 = 1.0 DNN unit); bands with `n_eff = (Σc)²/Σe² < --gof-neff-min`
(10) are merged with a neighbour; inside each band the cells are swept in HME
and accumulated into a super-cell until `n_eff ≥ --gof-neff-min`; all empty
observations form one extra term.

```
χ²_sc = Σ_sc (C − M)² / (M·w̄ + (thr·C)²) + M_e / w_glob
C = Σc,  M = Σμ̂,  w̄ = Σe²/Σc,   ndf = n_terms − npar
(n_terms = n_supercells, plus one when there is an empty observation)
p = TMath::Prob(χ²_sc, ndf)      PASS ⇔ converged ∧ covariance ∧ p ≥ 0.05
```

(a super-cell with `M ≤ 0` falls back to the observed variance
`Σe² + (thr·C)²`).  The −2lnL minimum is kept as `objective`, the deviance
calibration as `gof_deviance` (diagnostic only).

**Warm start (`--warm-start`).** Each thr step starts from the previous
converged solution; a new degree embeds the previous degree's best fit by
term name (new terms 0, same `N`).  The warm fit is kept if it converges,
otherwise the cold fit also runs and the lower objective wins (`start_kind`,
`objective_warm`, `objective_cold`).

## Presets

| `--preset` | `--norm` | `--stat` | warm start | acceptance GoF |
|------------|----------|----------|------------|----------------|
| `v1` (default) | `window_gl` | `chi2_neyman` | off | per-cell χ², `TMath::Prob(χ², n_obs − npar)` |
| `v2` | `tiling` | `poisson_eff` | on | super-cell Pearson χ² |

Explicit `--norm`, `--stat`, `--warm-start`/`--no-warm-start` override the
preset; any combination runs and the GoF follows `--stat`.  `v1` reproduces
the August-2026 published fits bit-for-bit and prints a warning about their
yield bias; every fit JSON carries a `caveats` list describing the known
deficiencies of its configuration (see *Outputs* and *Status*).

## Outputs

**Fit JSON** (`fit_one.py --output`).  Fields read by downstream code:

| Field | Meaning |
|-------|---------|
| `key`, `model`, `function` | `ExpPolyLogY2D-<d>`; `function` = {`model`, `norm`, `degree`, `param_names`, `terms`, `n_quad`, `y_eps`, `fit_mode`, `n_quad_fine`} — enough to rebuild the density (`core.rebuild_density_2d`) |
| `param_names`, `parameters`, `errors`, `covariance` | `N` first, then `a_ij`; covariance is `npar × npar` |
| `degree`, `thr_min` | selected polynomial degree and accepted relative uncertainty |
| `chi2`, `ndf`, `chi2ndf`, `p_value`, `converged`, `hesse_ok` | acceptance GoF (super-cell values for `poisson_eff`) and fit status |
| `sum_mu`, `sum_data`, `yield_closure` | Σμ over the observed cells, Σ MC content, and their ratio |
| `norm`, `stat`, `preset`, `warm_start`, `start_kind`, `objective`, `objective_warm`, `objective_cold` | fit configuration and −2lnL (or χ²) at the minimum (warm and cold attempts when both ran) |
| `caveats` | list of strings: known deficiencies of the configuration (Neyman χ² yield blindness and `--n-quad` dependence for `v1`; uncalibrated acceptance test for `v2`); copied into the templates' `meta_json` |
| `max_exp_arg`, `clip_hit` | largest exponent seen at the quadrature nodes and whether the ±50 clip was hit (density unreliable off the fine grid if true) |
| `fine_binning` | `x_edges`, `y_edges` of the fine grid (`tiling` only); every in-window fine bin belongs to an observation (a window that leaves bins uncovered or cuts a cell is an error) |
| `gof`, `gof_deviance`, `weight_map_stats`, `gof_supercells_build` | GoF details (`poisson_eff` only): `gof = {stat, chi2, ndf, chi2ndf, p, n_supercells, neff_min, band_nx, empty_term, supercells: [{band, ix0, ix1, ylo, yhi, n_cells, C, E2, M, pull, …}]}` |
| `fit_range`, `mx`, `category`, `process`, `grid_file`, `source_file`, `hist`, `trials` | provenance, fit window, per-trial scan record |

**Template file** (`make_hist_from_fit.py -o`).  One `TDirectory`
`m{MX}_{key}` per fit (or flat names `m{MX}_{key}_<hist>` with `--flat`)
containing `TH2D` histograms on the requested binning (`--from-hist` copies the
input TH2 edges; or `--x-range/--x-bins/--y-range/--y-bins`), integrated with
`--n-quad` (default 8) nodes per axis per bin:

| Histogram | Content |
|-----------|---------|
| `nominal` | `μ_i`, bin error `δ_i = thr_min · μ_i` |
| `nominal_acceptedUnc` | `δ_i` (absolute) |
| `nominal_acceptedUncUp`, `nominal_acceptedUncDown` | `μ_i ± δ_i` (Down floored at 0) |
| `nominal_eig{k}Up`, `nominal_eig{k}Down` (`--shape-variations`) | template at `p ± √λ_k · u_k` for each covariance eigenmode k (`N` frozen unless `--vary-yield`; `--max-modes` limits k) |
| `meta_json` (`TObjString`) | provenance: source JSON, binning, `norm`, `stat`, `preset`, `p_value`, `clip_hit`, `caveats`, `thr_min`, list of accepted-unc histograms |

`--thr` overrides `thr_min` for the accepted-unc histograms;
`--no-accepted-unc` drops them.  `--fit-dir` processes every fit JSON of a
directory into one file.

**Slice plots** (`plot_fit_slices.py`).  One page per DNN slice (or
`--pads-per-page`), DY MC points vs fit line as HME projections, legend with
slice integrals and their errors (fit error from the covariance eigenmodes),
global χ²/ndf and p-value.  Slices: `--auto-dnn-slice-n-const-signal-bins N`
(equal signal yield from `plots_2d/signal_2d`) or explicit
`--dnn-slices=-7.5,-5,-3,-1,0,1,2,3,6.5` (must match bin edges).

## Data layout

```
{data-dir}/hadd_m{MX}_res2b.root
  plots_2d/DY_2d        TH2D  x = DNN ∈ [−7.5, 7.5] (150 bins), y = HME ∈ [0, 2500] GeV (250 bins)
  plots_2d/signal_2d    TH2D  same binning (only for the equal-signal DNN slicing)
```

`grid_merge.py` reads one TH2 (`--input`, `--hist`); `plot_fit_slices.py`
locates the hadd file from `--data-dir` and the `mx`/`category` stored in the
fit JSON.  Grid and fit JSONs record the source file, histogram and grid
metadata.

## Status (2026-09-08)

- `--preset v1` reproduces the August-2026 published fits bit-for-bit.  Those
  fits carry a **yield bias**: Σμ/ΣMC per mass (`yield_closure`) ranges from
  0.29–0.97 over the ten masses.  Cause: the Neyman χ² with σ = the observed
  cell error on thousands of single-MC-event cells is blind to the yield, and
  the window-GL normalisation makes the template yield depend on `--n-quad`.
  **The published August templates must not be used for results.**
- `--preset v2` (tiling normalisation, Bohm–Zech scaled-Poisson likelihood
  with effective weights, super-cell Pearson χ² GoF, warm starts) restores the
  closure to 0.97–1.03, but the ExpPolyLogY2D family then fails the super-cell
  GoF for most masses at degree ≤ 8 and θ ≤ 0.10: the automatic scan passes
  2 of the 10 masses (mX = 650 and 700 GeV); for mX = 600 and 900 GeV the test
  runs out of degrees of freedom at degree 8 (fewer super-cells than
  parameters).  A first synthetic-toy study (mX = 500, 200 toys) indicates that
  the super-cell test over-rejects on every synthetic null tried (dominated by
  sparse cells, but also through the data-driven partition and the heavy-tailed
  weights), so the GoF verdicts are themselves provisional until the test is
  calibrated on a null that reproduces the MC weight mixture.
- Therefore the templates are **not production-ready**; the model family and
  the acceptance test are under study.  The thresholds (p ≥ 0.05, θ ≤ 0.10,
  degree ≤ 8), fit windows and the choice of category are analysis decisions
  and are not changed here.
