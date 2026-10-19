import os
import sys

# create_dataset.py moved to Studies/DNN_Dataset upstream
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DNN_Dataset"))
from create_dataset import add_weight_file

# Regenerates m600 weight files restricted to a single background, for the
# TT-only / DY-only ABCDisCo baseline runs. Reuses the existing merged
# dataset (nParity{j}_Merged.root) - does not rerun measure_cut_datasets/hadd.

dataset_folder = "/eos/user/t/thorntoa/DNNDatasets/ResolvedDataset_v2605a_Disco/Dataset"
mass = 600

# class_value from dataset_setup_doubleLep_resolved_v2605a.yaml: TT=1, DY=2, Other=3
add_weight_file(dataset_folder, mass=mass, keep_backgrounds=[1], suffix="_TTonly")
add_weight_file(dataset_folder, mass=mass, keep_backgrounds=[2], suffix="_DYonly")
