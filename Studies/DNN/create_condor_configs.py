import os
import yaml
import awkward as ak

resolved = 1
# Which background to train against: "all", "TTonly", "DYonly".
# TTonly/DYonly point at the restricted weight files made by
# make_background_only_weights.py (keep_backgrounds=[1]/[2]).
background_variant = "all"
if resolved:
    # Resolved ABCDisCo (binary BCE + dCorr against HME on background)
    template = "config/training_setup_doubleLep_resolved_disco.yaml"
    output_folder = "CondorConfigs/DoubleLepton_Resolved_Disco_v1"

    # Lambda scan: par0 only (one parity is enough to compare lambdas). Set to
    # range(4) for a full 4-parity production of the chosen lambda.
    parities = [0]

    input_file_template = "/eos/user/t/thorntoa/DNNDatasets/ResolvedDataset_v2605a_Disco/Dataset/nParity{j}_Merged.root"

    mass_specific = True
    mass_list = [600]
    weight_suffix = {"all": "", "TTonly": "_TTonly", "DYonly": "_DYonly"}[
        background_variant
    ]
    weight_file_template = (
        "/eos/user/t/thorntoa/DNNDatasets/ResolvedDataset_v2605a_Disco/Dataset/"
        f"nParity{{j}}_Merged_weight_m{{m}}{weight_suffix}.root"
    )

    training_name = "DNN_DoubleLepton_Resolved_Disco_Training{i}_par{j}_m{m}"
    var_parse_dict = {
        "learning_rate": [0.0005],
        "n_epochs": [10],
        "dropout": [0.2],
        # 'parametric_list': [ [ 600 ] ],
        "parametric_list": [[-1]],
        "l2_rate": [0.001],
        "UseParametric": [False],
        "use_batch_norm": [True],
        "nClasses": [1],
        # dCorr memory grows with batch_size squared, keep it moderate
        "batch_size": [2048],
        # Lambda scan: Training{i} -> lambda = scan_list[i] (5*i here)
        "disco_lambda_factor": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "patience": [50],
        "lr_patience": [3],
        "lr_decay": [0.8],
    }

else:
    # Boosted
    template = "config/training_setup_doubleLep_boosted.yaml"
    output_folder = "CondorConfigs_23Apr/DoubleLepton_Boosted_v3"

    parities = range(4)

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

        config = default_config.copy()
        for name, var in zip(var_names, varset.tolist()):
            if name == "parametric_list" and var == [-1]:
                print(f"Parametric list is empty, set to special mass {m}")
                var = [m]
            config[name] = var
        for j in parities:
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
            outFilePath = os.path.join(output_folder, outFileName)
            with open(outFilePath, "w") as f:
                yaml.dump(config, f)
