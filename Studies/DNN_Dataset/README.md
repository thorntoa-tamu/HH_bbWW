# DNN training datasets

`create_dataset.py` builds the training files for both DNN trainings, TensorFlow (`Studies/DNN`)
and PyTorch (`Studies/DNN_pytorch`), from HistTuples.

```sh
cd Studies/DNN_Dataset
# list the input files found per dataset and era, without processing anything
python3 create_dataset.py --config config/dataset_setup_doubleLep_v2610.yaml --dry-run
python3 create_dataset.py --config config/dataset_setup_doubleLep_v2610.yaml \
    --output-folder /eos/user/<u>/<you>/HH_bbWW/DNNDatasets/<tag>
```

Inputs on another site are read over xrootd, so a valid grid proxy is needed
(`voms-proxy-init -voms cms -rfc`).

## Config

| Key | Meaning |
|---|---|
| `storage_folders` | `{era: base}`. Each `base` is a local path or an xrootd URL (`root://cmseos.fnal.gov//store/...`), so eras can live on different sites. Files are read from `<base>/<dataset>/*.root`; a dataset missing in an era is skipped for that era. |
| `storage_folder` | Legacy single base; glob wildcards are allowed (`.../HistTuples/*/` for all eras). Used only when `storage_folders` is absent. |
| `nParity`, `parity_cut` | Events with `event % nParity == k` go to fold `k`. |
| `iterate_cut` | RDF filter applied before the split. |
| `extra_vars` | `false` (default): save all HistTuple columns plus `class_value` and `X_mass`. `true`: also run `add_extra_vars` (p4 components, etc.). |
| `weight_files.all_masses` | Write `nParity{k}_Merged_weight.root` (legacy PyTorch configs without a `weighting` block). No weight file is written when `weight_files` is absent. |
| `weight_files.per_mass_signal` | Signal key; writes `nParity{k}_Merged_weight_m{mass}.root` for each of its `mass_points`, where other masses get weight 0 (TensorFlow). `null` to skip. |
| `signal`, `background` | Datasets per process, and their `class_value`. Signals use `class_value <= 0`. |

## Outputs (under `<output-folder>/Dataset/`)

- `nParity{k}_Merged.root`: `Events` tree with the selected columns plus `class_value` and `X_mass`
  (0 for backgrounds).
- `nParity{k}_Merged_weight[_m{mass}].root`, only when `weight_files` asks for it, event-aligned with the file above:
  - `weight_tree` (TTree) with branches:
    - `class_target`: all signals set to 0;
    - `class_targets_binary`: 0 for signal, 1 for background;
    - `class_weight`: negative weights set to 0, clipped at mean + 3σ, total signal scaled to total background;
    - `multiclass_weight`: each background class scaled to the total background.
  - `weighted_class_targets` and `weighted_class_targets_multiclass`: TH1D of `class_value`, filled with those weights.
- `dataset_distribution_parity{k}.yaml`: per dataset, the `total` and `total_cut` events, the
  `total_cut_weighted` yield, and `eras` (the number of input files each era provided).
- `nParity{k}_Merged_input_features/`: input-feature plots per class.

`dataset_setup_doubleLep_merged_nocuts.yaml` with this script reproduces the `July25` dataset used
by the PyTorch v7 config (`DNNDatasets/July25`). That dataset was produced on 2026-07-27 by
commit `30deee4` of the unmerged `Update_HistFlavor` branch.
