# Parametric DNN (PyTorch)

Training and validation of the parametric multiclass DNN used in the dilepton channel. The
production models in `config/DNN/DoubleLepton_Parametric_{Resolved,Boosted}_v1/` were trained
here with `model_configs/looseBTag_multiclass_Radion_pDNN_v6.yaml`; their `dnn_config.yaml` is
that file, carried over to the deployed schema (see [Config schema](#config-schema)).

One network is trained per regime (`resolved`, `boosted`), with the signal mass hypothesis
`X_mass` as an input. Mass-dependent features (`ll_dR_scaled`, `bb_dR_scaled`, …) are recomputed
for each hypothesis; background events are assigned a hypothesis drawn from the signal mass
points. With `spin` as an input the model is also parametric in spin: signal events carry their
own spin and background events get a random one, like the mass.

## Inputs

The training datasets are the `nParity{0..3}_Merged.root` files written by
`Studies/DNN_Dataset/create_dataset.py` (see its [README](../DNN_Dataset/README.md)). Configs with a
`weighting` block compute the event weights from `weight_Central`; older configs read them from
the `nParity{0..3}_Merged_weight.root` files instead.

```sh
cd Studies/DNN_Dataset
python3 create_dataset.py --config config/dataset_setup_doubleLep_merged_nocuts.yaml \
    --output-folder /eos/user/<u>/<you>/HH_bbWW/DNNDatasets/<tag>
```

`dataset_setup_doubleLep_merged_nocuts.yaml` is the setup of the `July25` dataset used by the v7
config (identical to the copy stored with that dataset). The setup of the `July24` dataset used
by v6 was not kept. `dataset_setup_doubleLep_v2610.yaml` is the setup of the v2610 dataset
(`/eos/user/d/daebi/HH_bbWW/DNNDatasets/v2610_training/Dataset`) used by the `v2610_pDNN_v1*` configs.

## Running

Run from this directory (the scripts import `src.model_helper` relative to it):

```sh
python3 train_pdnn.py --config model_configs/v2610_pDNN_v1.yaml
python3 apply_pdnn.py --config model_configs/v2610_pDNN_v1.yaml
python3 limit_pdnn.py --config model_configs/v2610_pDNN_v1.yaml
python3 limit_pdnn.py --compare <output_folder A>/limits/limits.json <output_folder B>/limits/limits.json \
    --labels A B --output-dir <plots dir>
python3 shape.py
```

`train_pdnn.py` takes `--folds` (parity cycles to run, e.g. one per batch job), `--regimes` and
`--epochs` to run part of a training; the default is every fold and regime of the config.
`--threads` (default 4) caps the CPU threads: on shared lxplus nodes more threads run slower.

### GPU (lxplus-gpu)

The torch of the analysis environment is a CPU-only build; `run_lxplus_gpu.sh` sets up the CUDA LCG
view and runs `train_pdnn.py` with the given arguments. On a T4 an epoch of the resolved network
takes about 10 s, against about 4 minutes on an lxplus CPU node; one fold of one model takes about
30 minutes, most of it reading the dataset.

```sh
ssh lxplus-gpu.cern.ch          # note the node, to come back to the tmux session
tmux new -s pdnn
cd <repo>/Studies/DNN_pytorch
./run_lxplus_gpu.sh --config model_configs/v2610_pDNN_v1.yaml --folds 0
```

The folds can run as separate jobs on different nodes. `limit_pdnn.py` and `overtraining_check.py`
need the analysis environment (`source env.sh`), for `combine` and `onnxruntime`.

| Script | Does | Writes (under `output_folder`) |
|---|---|---|
| `train_pdnn.py` | Trains one model per regime and parity fold, exports ONNX, evaluates on the validation fold | `nParity{f}_validation/pdnn_model_{regime}_nparity{f}.onnx`, `validation_applied.root` (signal log-odds `DNN_M{m}` and class probabilities per hypothesis), `loss_history_{regime}.json` / `loss_curve_{regime}.png` |
| `apply_pdnn.py` | Re-evaluates the exported ONNX models on the validation folds | `nParity{f}_validation/validation_applied_onnx.root` |
| `overtraining_check.py` | Scores each cycle's ONNX models on their training parity and on the validation parity, at a few masses: per-class weighted KS tests and signal-vs-background AUC, train vs validation | `overtraining/overtraining_check.pdf`, `.json` |
| `feature_importance.py` | Permutation importance of a trained model on each cycle's validation parity: each input, and each group of inputs correlated with \|r\| > 0.8, shuffled at several masses; drop in signal-vs-background AUC (relative to 1 - AUC) and rise in class-balanced loss, averaged over cycles. Runs on the GPU when available (`--folds` to split cycles over jobs, `--summarize` to combine them) | `feature_importance/importance_{regime}.csv`, `.png`, `importance_vs_mass_{regime}.png` |
| `score_deployed.py` | Scores a training dataset with a deployed model (`config/DNN/<version>`, fold offsets as in `Analysis/DNN_Application.py`) into the `validation_applied.root` layout, so `limit_pdnn.py` can compare it with new models | `nParity{k}_validation/validation_applied.root` under the `--config` output folder |
| `production_comparison.py` | Projects full-analysis expected limits with the ratio of quick limits of two models, and plots them with the Run 2 expected limits | `production_comparison_{category}.png`, `.csv` |
| `limit_pdnn.py` | Expected limits per mass and spin from the validation trees of all folds: pDNN-logit templates in res2b / recovery / boosted, stat-only `combine -M AsymptoticLimits -t -1` per category and combined; `--compare` overlays models | `limits/limits.json`, `limits/limits_{cat}.png`, cards and shapes in `limits/spin{s}_M{m}/` |
| `shape.py` | Combines the folds, bins DeepHME mass in pDNN slices, writes templates and stack plots | `combined_validation/shape_{cat}.root`, `pdnn_mass_stackplots/` |

`shape.py` takes no arguments: the input folder and binning are the constants at the top of the
file.

## Configs

| Config | Classes | Inputs | Dataset |
|---|---|---|---|
| `looseBTag_multiclass_Radion_pDNN_v6.yaml` | Signal, TT, Other | 21 common + regime features | `../Datasets/July24` |
| `looseBTag_multiclass_Radion_pDNN_v7.yaml` | Signal, TT, DY, ST, Other | v6 + `DeepHME_mass` | `July25` on EOS |
| `v2610_pDNN_v1.yaml` | Signal, TT, DY, Other | v6 without `DeepHME_mass`, + `MX_reduced`, `Pzeta`, `MT2_bb`, `MT2_ll`, `DeepHME_mass_error_rel` | v2610 on EOS |
| `v2610_pDNN_v1_spin.yaml` | Signal, TT, DY, Other | v2610_pDNN_v1 + `spin` | v2610 on EOS |

v6 is the production model. v7 is not deployed, and does not train five classes as written: the
network gets `len(class_names)` outputs, but `load_parametric_fold` assigns only three targets
(0 signal, 1 TT, 2 everything else), so the DY and ST outputs are never a target and output 2,
labelled DY, holds all non-TT background.

### Signal, classes and weights (v2610 configs)

| Key | Meaning |
|---|---|
| `signal_classes` | Dataset `class_value`s trained as signal. Default: every `class_value <= 0`. |
| `output_classes` | Ordered `{output name: signal \| [class_values] \| rest}`; the first must be `signal`, `rest` takes every background not listed. Default: Signal / TT / Other. Events with no output (e.g. 1L signals) are not trained on but are still scored in the validation tree. |
| `signal_spin`, `spin_values` | Spin of each signal `class_value`, and the spin hypotheses drawn for backgrounds when `spin` is an input. Validation scores are then written per spin, `DNN_M{m}_S{spin}`. |
| `weighting` | Weights from `weight_Central`: `zero_negative`, `clip_sigma` (clip at mean + N sigma per loaded fold), `flatten_signal_per_mass` (equal total weight per signal mass in each regime) with `flatten_min_events` (masses with fewer signal events, e.g. boosted 250-400 GeV, keep their own weight). After this every output class is scaled to the signal total. |

`DeepHME_mass_error_rel` (`DeepHME_mass_error / DeepHME_mass`) is computed by the loader; the
deployed `Analysis/DNN_Application.py` does not compute it yet.

## Folds and config schema

`create_dataset.py` writes the events with `event % nParity == k` to `nParity{k}_Merged.root`.
In cycle `i`, `train_pdnn.py` trains on file `index(train_parity)`, uses file
`index(test_parity)` for the LR schedule, early stopping and the choice of the saved epoch, and
exports the model as `nparity{i}`. Only the `index` expressions are read; `func` is not used by
training.

The deployed `dnn_config.yaml` files use the schema `Analysis/DNN_Application.py` reads — an
integer `offset` per split, and `model_name` formatted with `{fold}` — so a model copied into
`config/DNN/` needs its config converted by hand: the `index` expression `(nParity + k) % 4`
becomes `offset: k`. Fold `f` is applied to `event % nParity == (f + offset) % nParity`, the same
bucket `index` gives.

`apply_pdnn.py` and `limit_pdnn.py` read the fold of each cycle from the `train_parity` and
`val_parity` `index` expressions.
