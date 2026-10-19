# Running the analysis

The pipeline — anaTuples, observables, histograms, plots — is the **standard FLAF chain**; follow
the **[FLAF full-workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)** for
the commands. This page collects what is **specific to HH→bb̄WW**.

```sh
ERA=Run3_2022
VER=dev

law run FLAF.Analysis.tasks.HistPlotTask --era $ERA --version $VER --workflow local --test 1000
```

## The b-tag-shape cache runs first

HH→bb̄WW computes **b-tag shape** weights in a dedicated caching stage,
`AnalysisCacheTask` (aggregated by `AnalysisCacheAggregationTask`), which LAW runs automatically
before histogramming — see
[FLAF → Task reference](https://cms-flaf.github.io/FLAF/reference/tasks/#analysiscachetask).

!!! warning "Budget time for the cache on a cold start"
    On a cold cache `AnalysisCacheTask` is **slow** (roughly an hour per branch), even for simple
    variables. When iterating, reuse an existing cache instead of recomputing it, using a
    [per-task version override](https://cms-flaf.github.io/FLAF/workflow/arguments/#per-task-version-overrides),
    e.g. `--AnalysisCacheTask-version <existing> --AnalysisCacheAggregationTask-version <existing>`.

## Mass reconstruction: DeepHME

The HH mass is reconstructed with **DeepHME** (the bb̄WW counterpart to SVfit in bb̄ττ). It is part
of the observable computation and requires no special command — it runs as part of the standard
producer chain.

## Stored gen-level t̄t information

Processes with `genInfo: [ TT ]` store, next to the usual observables:

| Column | Content | Read by |
|---|---|---|
| `TTInfo_nLeptonicW` | number of leptonically decaying W bosons (0, 1 or 2; τ counts as leptonic) | — (t̄t is not stitched: `TT` uses only the three decay-channel samples) |
| `genTop_{pt,eta,phi,mass}` | last-copy top and anti-top, in this order | top-p<sub>T</sub> reweighting at `AnaTupleFileTask` (`genTop_pt`; configured for Run3_2022–2023BPix) |
| `genTop_b_{pt,eta,phi}`, `genTop_lep_{pt,eta,phi,mass}` | the b quark and the W's charged lepton of each top | — |
| `genTop_lep_gen_kind` | `GenLepton::Kind` of that lepton, -1 for a hadronic W | — |

`TTInfo_*` columns must stay scalars: the anaTuple stores all columns that share a prefix as one
collection, so an array under `TTInfo_` would turn the count into an array as well.

## Corrections in Run3_2024 and later

`config/Run3_2024/global.yaml` (and 2025, 2026) inherits the analysis-wide `corrections:` block
through the anchor `*corrections_default` and overrides only `btag` (UParTAK4, no shape
calibration yet) and `dy_hhbbtautau`. A correction added to `config/global.yaml` — the shape
weights `pu`, `parton_shower`, `top_pt`, `pdf` and `qcd_scale` included — therefore applies to
those eras too; an entry that has to differ there is overridden as a whole.

## Resonant mass grid

The bb̄WW signal is `GluGluTo{Radion,BulkGraviton}` at 40 mass points from 250 GeV to 5 TeV in each
of `Run3_2022` through `Run3_2023BPix`, with the single-lepton (`2B2JLNu`) and dilepton (`2B2L2Nu`)
final states as separate datasets. In 2023 and 2023BPix the whole grid comes from the custom
production (DSProd, read from private storage with `fs_nanoAOD:` + `dirName:`); in 2022 and 2022EE
18 points per final state do and the other 22 are central `Run3Summer22` datasets. The central
X→YH samples are not selected. The bb̄ττ signal (`2B2Tau`, processes
`GluGluTo{Radion,BulkGraviton}_bbTauTau`) has the same 40 masses: in 2023 and 2023BPix 29 points
per spin come from the custom production and 11 are central, in 2022 and 2022EE 7 per spin come
from the custom production and 33 are central. `Run3_Model` carries every hypothesis through
anaTuple and histogram production; a fit picks one later, in the datacard. See the
[physics model](setup.md#production-model).

Two networks consume that grid, and they do not cover the same part of it:

- the **dilepton DNN** is parametric — one model per fold, evaluated at whatever masses
  `DNN.columns` in `config/global.yaml` asks for. It is configured over the produced points up to
  1.5 TeV, and the datacard `param_values` in `config/Datacards/x_hh_bbww_{DL,SL}_run3.yaml` follow
  the same 25 points. The four lowest of them, 250 to 280 GeV, are below the lowest
  `signal_mass_points` of the models currently in `config/DNN/DoubleLepton_Parametric_*_v1`, so
  those scores extrapolate and stay provisional until the network is retrained over this grid.
- the **single-lepton two-stage DNN** carries one trained model per mass point in
  `config/DNN/SingleLepton_IndependentMasses/`, so its grid, and the `masspoints:` list of SL signal
  regions built from its scores, stay at the ten points those models were trained for until
  retrained ones exist.

Every mass point named in a `variables:` list also needs its binning in `config/plot/histograms.yaml`
— a variable with no matching entry there stops `HistProducerFromNTuple` with a `KeyError`.

## Columns taken from the central tree

`config/global.yaml` lists in `anaTuple_shift_invariant_columns` the anaTuple columns that no
systematic shift changes: event numbers and dataset metadata, generator weights, the pileup, parton-
shower, PDF, scale and top-p<sub>T</sub> weights, pileup truth, the LHE record and the generator truth
of the signal (`gen*`) and of t̄t (`TTInfo_*`, `genTop_*`). FLAF stores them in the central tree only
and fills them in for events that only a shift selected, after checking that every variation agrees.

- Only event-level generator quantities belong there. Generator information attached to a
  reconstructed object — `lepN_gen_*`, `centralJet_matchedGenJet_*`, the jet flavour labels — follows
  the selected object and changes under JES, JER and the lepton scales; the fuse step stops with
  `Column '…' is declared shift-invariant but differs in …` if one is listed.
- `HLT_*` is left out: `Analysis/hh_bbww.py` defines a trigger column that `GetColumnNames()` does not
  list, and a column taken from the central tree is not listed there.
- Changing the list changes the anaTuple layout, so it goes with a new anaTuple production.

## Categories

HH→bb̄WW is analysed in **resolved** and **boosted** categories (low- and high-p_T HH topologies).
Category and channel selection is driven by `config/global.yaml`; narrow or extend it there or via
your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/).

## Choosing which variables to histogram

As for any FLAF analysis, the variable set is controlled by the `variables:` list in
`user_custom.yaml` (or `--variables`). A short list keeps test runs fast:

```yaml
variables:
  - lep1_pt
  - ggF_DNN_HH
```

## Statistical interpretation

Continue to [Statistical inference](stat_inference.md) for datacards, limits and diagnostics.
