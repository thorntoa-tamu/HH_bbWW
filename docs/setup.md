# Setup

HH→bb̄WW is installed and run like any FLAF analysis. The general procedure — prerequisites (CERN
account, CVMFS, grid certificate, SSH keys) and what the first `source env.sh` builds — is in the
**[FLAF installation guide](https://cms-flaf.github.io/FLAF/getting-started/installation/)**. This
page lists the HH→bb̄WW specifics.

## Clone (with git LFS)

```sh
git clone --recursive git@github.com:cms-flaf/HH_bbWW.git
cd HH_bbWW
git lfs pull
source env.sh
```

!!! danger "Two easy-to-miss steps"
    - `--recursive` — pulls the submodules (without it, imports fail on empty directories).
    - `git lfs pull` — fetches the large files HH→bb̄WW tracks with git LFS. Skipping it leaves
      pointer placeholders instead of real files.

    To run a specific central production, clone its tag, e.g. `git clone -b <version> --recursive …`.

## Analysis-specific submodules

| Submodule | Role |
|---|---|
| `DeepHME` | Neural reconstruction of the HH mass for bb̄WW. |
| `SyncTool` | Synchronisation/validation tooling. |
| `StatInference`, `inference` | Datacards and combine-based limits/fits (shared). |

These build automatically as part of `source env.sh`; you do not set them up by hand.

## Production model

The default [physics model](https://cms-flaf.github.io/FLAF/configuration/processes-and-models/)
for HH→bb̄WW is `Run3_Model` (set in `config/global.yaml`). It carries every X→HH signal hypothesis
together — Radion and BulkGraviton, bb̄WW single lepton and dilepton, and bb̄ττ — and does not
select the central X→YH samples. AnaTuple and histogram production branch per dataset, and each merged histogram is
named by its expanded process (`GluGluToRadion_bbWW_2L_300`,
`GluGluToBulkGraviton_bbWW_1L_1000`, …), so the hypotheses are not added into one histogram. Which hypothesis a fit uses is chosen later, by the
datacard's process list; the dilepton card fits `GluGluToRadion_bbTauTau` together with
`GluGluToRadion_bbWW_2L`. A signal added to a physics model must also be registered in
`AnaProd/anaTupleDef.py`: in `signals_with_hvv` when its decay has an H→VV, whose gen-level
candidate the anaTuple then stores, or in `signals_without_hvv` otherwise. The AnaTuple job of
an unregistered signal stops at start-up. The samples cover Run3_2022 through Run3_2023BPix; later eras have no
resonant sample produced yet, so the model loads there with no signal in it. AnaTuple production
uses this same model (`phys_model: Run3_Model` in `config/user_custom.yaml`; see the repository
README). The single-Higgs background group is `H` (ggH, VBFH, VH and tt̄H summed into that
one histogram). For fast local tests, use `TestModel`
in your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/) instead.
It holds three backgrounds — `custom_CI_Background_TT`, one t̄t dataset (t̄t is not stitched),
`custom_CI_Background_DY` and `custom_CI_Background_W`, one DY and one W dataset carrying the
stitchers the era configures — plus one signal and one data process. Each CI background mirrors the `processors:` of the real
process it stands for, so the [stitching](https://cms-flaf.github.io/FLAF/concepts/stitching/)
is exercised over the whole anaTuple → merge chain; keep them in step when you change the real
ones.

## Next

- [Running the analysis](analysis.md) — the bb̄WW-specific run notes (DeepHME, b-tag-shape cache,
  categories).
- [FLAF → Full workflow](https://cms-flaf.github.io/FLAF/workflow/walkthrough/) — the common
  pipeline, stage by stage.
- [FLAF → HTCondor](https://cms-flaf.github.io/FLAF/workflow/htcondor/) /
  [CRAB](https://cms-flaf.github.io/FLAF/workflow/crab/) — CERN batch and full WLCG submission.
