# HH_bbWW

## How to run anaTuple production

Current version `v2610`

1. Clone repository
   ```bash
   git clone -b v2610 --recursive git@github.com:cms-flaf/HH_bbWW.git
   cd HH_bbWW
   git lfs pull
   source $PWD/env.sh
   law index
   ```

1. Define `config/user_custom.yaml` file as following:
   ```yaml
   fs_default: YOUR CERNBOX # davs://eoshome-k.cern.ch:8444/eos/user/k/kandroso/HH_bbWW/
   fs_anaTuple: root://cmseos.fnal.gov//eos/uscms/store/user/lpcflaf/HH_bbWW/

   phys_model: Run3_Model
   analysis_config_area: config
   compute_unc_variations: true
   compute_unc_histograms: true
   store_noncentral: true
   ```

1. Login to cms-flaf.cern.ch, enter screen session, login to lxplus
   ```bash
   ssh USER@cms-flaf.cern.ch
   screen -S HH_bbWW_production
   ssh lxplus.cern.ch
   kinit
   aklog
   ```

1. Load environment and setup grid certificate
   ```bash
   source $PWD/env.sh
   voms-proxy-init --voms cms --valid 192:00
   ```

1. Run production
   ```bash
   law run AnaTupleMergeTask --version v2610 --era ERA --parallel-jobs 2000 --bundle
   ```
   `AnaTupleFileTask` composes its jobs from the estimated cost of each input file; passing
   `--AnaTupleFileTask-tasks-per-job` turns that off and falls back to fixed-size groups.
