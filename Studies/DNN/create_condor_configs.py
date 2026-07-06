import os
import yaml
import awkward as ak

_script_dir = os.path.dirname(os.path.abspath(__file__))

resolved = 1
if resolved:
    # Resolved
    template = os.path.join(
        _script_dir, "config/training_setup_doubleLep_resolved.yaml"
    )
    output_folder = os.path.join(
        _script_dir, "CondorConfigs/DoubleLepton_IndependentMasses_v1"
    )

    # input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_Apr25/Dataset/nParity{j}_Merged.root"
    # input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_Apr16/Dataset/nParity{j}_Merged.root"
    input_file_template = (
        "/eos/user/t/thorntoa/DoubleLepton_v5/Dataset_Run3_2022/batchfile{j}.root"
    )

    mass_specific = True
    mass_list = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000]
    # weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_Apr25/Dataset/nParity{j}_Merged_weight_m{m}.root"
    # weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_Apr16/Dataset/nParity{j}_Merged_weight_m{m}.root"
    weight_file_template = (
        "/eos/user/t/thorntoa/DoubleLepton_v5/Dataset_Run3_2022/weightfile{j}.root"
    )

    training_name = "DNN_DoubleLepton_Resolved_Training{i}_par{j}_m{m}"
    var_parse_dict = {
        "learning_rate": [0.0005],
        "n_epochs": [5],
        "dropout": [0.2],
        # 'parametric_list': [ [ 600 ] ],
        "parametric_list": [[-1]],
        "l2_rate": [0.001],
        "gamma1": [3.0],
        "gamma2": [1.0],
        "n_layers": [3],
        "n_units_reduction_factor": [0.8],
        "loss_scale": [1.0],
        "UseParametric": [False],
        "use_batch_norm": [True],
        "nClasses": [2],
        "patience": [50],
        "lr_patience": [3],
        "lr_decay": [0.8],
        "disco_lambda_factor": [10],
    }

else:
    # Boosted
    template = os.path.join(_script_dir, "config/training_setup_doubleLep_boosted.yaml")
    output_folder = os.path.join(_script_dir, "CondorConfigs/DoubleLepton_Boosted_v1")

    # input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/BoostedDataset_Apr23/Dataset/nParity{j}_Merged.root"
    input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/BoostedDataset_Apr16_v4/Dataset/nParity{j}_Merged.root"

    mass_specific = True
    mass_list = [
        300,
        400,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
    ]
    # weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/BoostedDataset_Apr23/Dataset/nParity{j}_Merged_weight_m{m}_multiclass_noNegatives.root"
    weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/BoostedDataset_Apr16_v4/Dataset/nParity{j}_Merged_weight_m{m}.root"

    training_name = "DNN_DoubleLepton_Boosted_Training{i}_par{j}_m{m}"
    var_parse_dict = {
        "learning_rate": [0.00005],  # Frozen 0.005
        "n_epochs": [100],  # Frozen 100
        "dropout": [0.2],  # Frozen 0.2
        "parametric_list": [[-1]],
        # 'parametric_list': [ [ 300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000 ] ],
        "l2_rate": [0.001],  # Frozen 0.001
        "gamma1": [1.5],  # Frozen 1.5
        "gamma2": [0.9],  # Frozen 0.9
        "n_layers": [3],  # Frozen 3
        "n_units_reduction_factor": [1.0],  # Frozen 1
        "loss_scale": [0.8],  # Frozen 0.5
        "UseParametric": [False],
        "use_batch_norm": [True],
        "nClasses": [2],
        "patience": [50],
        "lr_patience": [3],
        "lr_decay": [0.8],
    }

os.makedirs(output_folder, exist_ok=True)

with open(template, "r") as f:
    default_config = yaml.safe_load(f)


var_names = var_parse_dict.keys()
var_combinations_list = [x for x in var_parse_dict.values()]
var_combinations = ak.cartesian(var_combinations_list, axis=0)

for i, varset in enumerate(var_combinations):

    for m in mass_list:

        mass_folder = os.path.join(output_folder, f"m{m}")
        os.makedirs(mass_folder, exist_ok=True)

        config = default_config.copy()
        for name, var in zip(var_names, varset.tolist()):
            if name == "parametric_list" and var == [-1]:
                print(f"Parametric list is empty, set to special mass {m}")
                var = [m]
            config[name] = var
        for j in range(4):
            # Set up each parity
            config["training_file"] = input_file_template.format(j=j)
            config["weight_file"] = weight_file_template.format(j=j, m=m)
            config["test_training_file"] = input_file_template.format(j=(j + 1) % 4)
            config["test_weight_file"] = weight_file_template.format(j=(j + 1) % 4, m=m)
            config["validation_file"] = input_file_template.format(j=(j + 2) % 4)
            config["validation_weight_file"] = weight_file_template.format(
                j=(j + 2) % 4, m=m
            )

            config["training_name"] = training_name.format(i=i, j=j, m=m)

            outFileName = f"{training_name.format(i = i, j = j, m = m)}.yaml"
            outFilePath = os.path.join(mass_folder, outFileName)
            with open(outFilePath, "w") as f:
                yaml.dump(config, f)
