#!/bin/bash
# Runs train_pdnn.py on a GPU node (lxplus-gpu, or an HTCondor GPU slot).
# The torch of the analysis environment (soft/flaf_env) is a CPU-only build, so this uses the
# CUDA LCG view instead; train_pdnn.py only needs torch, uproot, pandas, scikit-learn, onnxruntime.
#
#   ssh lxplus-gpu.cern.ch
#   cd <repo>/Studies/DNN_pytorch
#   ./run_lxplus_gpu.sh --config model_configs/v2610_pDNN_v1.yaml [--folds 0 1 ...] [--regimes ...]
#
# Interactive sessions end when you log out: start it inside tmux on the same node.

LCG_VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt}
source "$LCG_VIEW/setup.sh"
cd "$(dirname "$0")"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
python3 -u train_pdnn.py "$@"
