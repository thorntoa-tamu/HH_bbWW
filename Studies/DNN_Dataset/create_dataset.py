import os
import uproot
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()
ROOT.EnableImplicitMT(4)


log_variables = [
    # "lep1_pt",
    # "lep2_pt",
    # "PuppiMET_pt",
    # "HT",
    # "MT",
    # "MT2_ll",
    # "MT2_bb",
    # "MT2_blbl",
    # "MT2_blbl2",
    # "ll_mass",
    # "bjet1_pt",
    # "bjet1_mass",
    # "bjet2_pt",
    # "bjet2_mass",
    # "wjet1_pt",
    # "wjet1_mass",
    # "wjet2_pt",
    # "wjet2_mass",
    # "fatbjet_pt",
    # "fatbjet_mass_PNetCorr",
    # "DoubleLep_DeepHME_mass",
]


def add_extra_vars(rdf_tmp, class_value, X_mass):
    rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
    rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
    rdf_tmp = rdf_tmp.Define("lep1_legType", "int(channelId/10.0)")
    rdf_tmp = rdf_tmp.Define("lep2_legType", "int(channelId%10)")
    rdf_tmp = rdf_tmp.Define(
        "DeepHME_mass_error_rel",
        "float(DeepHME_mass_error)/float(DeepHME_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "b1_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet1_pt, bjet1_eta, bjet1_phi, bjet1_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "b2_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet2_pt, bjet2_eta, bjet2_phi, bjet2_mass)",
    )
    # other_jet1/2 branches do not exist in v2605a (wjet1/2 replace them but are
    # only filled for SL events), so no j1_p4/j2_p4 here
    rdf_tmp = rdf_tmp.Define(
        "fatjet_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(fatbjet_pt, fatbjet_eta, fatbjet_phi, fatbjet_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "lep1_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep1_pt, lep1_eta, lep1_phi, lep1_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "lep2_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep2_pt, lep2_eta, lep2_phi, lep2_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "met_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(PuppiMET_pt, 0, PuppiMET_phi, 0)",
    )

    rdf_tmp = rdf_tmp.Define(
        "dR_b1leps",
        "TMath::Min(ROOT::Math::VectorUtil::DeltaR(b1_p4, lep1_p4), ROOT::Math::VectorUtil::DeltaR(b1_p4, lep2_p4))",
    )
    rdf_tmp = rdf_tmp.Define(
        "dR_b2leps",
        "TMath::Min(ROOT::Math::VectorUtil::DeltaR(b2_p4, lep1_p4), ROOT::Math::VectorUtil::DeltaR(b2_p4, lep2_p4))",
    )

    rdf_tmp = rdf_tmp.Define(
        "m_b1leps",
        "ROOT::Math::VectorUtil::DeltaR(b1_p4, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(b1_p4, lep2_p4) ? (b1_p4 + lep1_p4).M() : (b1_p4 + lep2_p4).M()",
    )
    rdf_tmp = rdf_tmp.Define(
        "m_b2leps",
        "ROOT::Math::VectorUtil::DeltaR(b2_p4, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(b2_p4, lep2_p4) ? (b2_p4 + lep1_p4).M() : (b2_p4 + lep2_p4).M()",
    )

    # rdf_tmp = rdf_tmp.Define("m_b1l1", "(b1_p4 + lep1_p4).M()")
    # rdf_tmp = rdf_tmp.Define("m_b1l2", "(b1_p4 + lep2_p4).M()")
    # rdf_tmp = rdf_tmp.Define("m_b2l1", "(b2_p4 + lep1_p4).M()")
    # rdf_tmp = rdf_tmp.Define("m_b2l2", "(b2_p4 + lep2_p4).M()")

    # rdf_tmp = rdf_tmp.Define("dR_b1l1", "ROOT::Math::VectorUtil::DeltaR(b1_p4, lep1_p4)")
    # rdf_tmp = rdf_tmp.Define("dR_b1l2", "ROOT::Math::VectorUtil::DeltaR(b1_p4, lep2_p4)")
    # rdf_tmp = rdf_tmp.Define("dR_b2l1", "ROOT::Math::VectorUtil::DeltaR(b2_p4, lep1_p4)")
    # rdf_tmp = rdf_tmp.Define("dR_b2l2", "ROOT::Math::VectorUtil::DeltaR(b2_p4, lep2_p4)")

    rdf_tmp = rdf_tmp.Define("pt_ll", "(lep1_p4 + lep2_p4).Pt()")
    rdf_tmp = rdf_tmp.Define("pt_bb", "(b1_p4 + b2_p4).Pt()")
    rdf_tmp = rdf_tmp.Define("m_llmet", "(lep1_p4 + lep2_p4 + met_p4).M()")
    rdf_tmp = rdf_tmp.Define(
        "m_bbllmet", "(b1_p4 + b2_p4 + lep1_p4 + lep2_p4 + met_p4).M()"
    )

    # Begin Run2 block
    rdf_tmp = rdf_tmp.Define("lep1_E", "(lep1_p4).E()")
    rdf_tmp = rdf_tmp.Define("lep1_px", "(lep1_p4).px()")
    rdf_tmp = rdf_tmp.Define("lep1_py", "(lep1_p4).py()")
    rdf_tmp = rdf_tmp.Define("lep1_pz", "(lep1_p4).pz()")

    rdf_tmp = rdf_tmp.Define("lep2_E", "(lep2_p4).E()")
    rdf_tmp = rdf_tmp.Define("lep2_px", "(lep2_p4).px()")
    rdf_tmp = rdf_tmp.Define("lep2_py", "(lep2_p4).py()")
    rdf_tmp = rdf_tmp.Define("lep2_pz", "(lep2_p4).pz()")

    rdf_tmp = rdf_tmp.Define("bjet1_E", "(b1_p4).E()")
    rdf_tmp = rdf_tmp.Define("bjet1_px", "(b1_p4).px()")
    rdf_tmp = rdf_tmp.Define("bjet1_py", "(b1_p4).py()")
    rdf_tmp = rdf_tmp.Define("bjet1_pz", "(b1_p4).pz()")

    rdf_tmp = rdf_tmp.Define("bjet2_E", "(b2_p4).E()")
    rdf_tmp = rdf_tmp.Define("bjet2_px", "(b2_p4).px()")
    rdf_tmp = rdf_tmp.Define("bjet2_py", "(b2_p4).py()")
    rdf_tmp = rdf_tmp.Define("bjet2_pz", "(b2_p4).pz()")

    rdf_tmp = rdf_tmp.Define("fatjet_E", "(fatjet_p4).E()")
    rdf_tmp = rdf_tmp.Define("fatjet_px", "(fatjet_p4).px()")
    rdf_tmp = rdf_tmp.Define("fatjet_py", "(fatjet_p4).py()")
    rdf_tmp = rdf_tmp.Define("fatjet_pz", "(fatjet_p4).pz()")

    rdf_tmp = rdf_tmp.Define("met_E", "(met_p4).E()")
    rdf_tmp = rdf_tmp.Define("met_px", "(met_p4).px()")
    rdf_tmp = rdf_tmp.Define("met_py", "(met_p4).py()")
    rdf_tmp = rdf_tmp.Define("met_pz", "(met_p4).pz()")

    for var in log_variables:
        rdf_tmp = rdf_tmp.Define(
            f"{var}_log", f"TMath::Log(({var} > 0 ? {var} : 0) + 1.0)"
        )

    cols_to_save = [col for col in rdf_tmp.GetColumnNames() if not col.endswith("_p4")]

    return rdf_tmp, cols_to_save


def get_storage_folders(config_dict):
    # storage_folders: {era: base} where base is a local path or an xrootd URL
    #   (root://host//store/...), so eras can live on different sites.
    # storage_folder (legacy): one base, glob wildcards allowed (e.g. .../HistTuples/*/)
    if "storage_folders" in config_dict:
        return config_dict["storage_folders"]
    return {"all": config_dict["storage_folder"]}


def list_root_files(base, dataset_name):
    if base.startswith("root://"):
        host, _, path = base[len("root://") :].partition("/")
        server = f"root://{host}"
        path = os.path.join("/" + path.lstrip("/"), dataset_name)
        result = subprocess.run(
            ["xrdfs", server, "ls", path], capture_output=True, text=True
        )
        if result.returncode != 0:
            return []
        return sorted(
            f"{server}/{p}" for p in result.stdout.split() if p.endswith(".root")
        )
    return sorted(glob.glob(os.path.join(base, dataset_name, "*.root")))


def collect_inputs(storage_folders, dataset_name):
    files = []
    eras_found = {}
    for era, base in storage_folders.items():
        era_files = list_root_files(base, dataset_name)
        if len(era_files) > 0:
            files += era_files
            eras_found[era] = len(era_files)
    return files, eras_found


def get_dataset_list(config_dict):
    # (process_name, key in the distribution yaml, dataset_name, class_value, X_mass)
    datasets = []
    for signal_name, signal_dict in config_dict["signal"].items():
        for mass_point in signal_dict["mass_points"]:
            dataset_name = signal_dict["dataset_name_format"].format(mass_point)
            datasets.append(
                (
                    signal_name,
                    mass_point,
                    dataset_name,
                    signal_dict["class_value"],
                    mass_point,
                )
            )
    for background_name, background_dict in config_dict["background"].items():
        for dataset_name in background_dict["background_datasets"]:
            datasets.append(
                (
                    background_name,
                    dataset_name,
                    dataset_name,
                    background_dict["class_value"],
                    0,
                )
            )
    return datasets


def print_input_summary(storage_folders, inputs):
    eras = list(storage_folders.keys())
    name_width = max(len(dataset_name) for dataset_name in inputs)
    print(f"{'dataset':<{name_width}} " + " ".join(f"{era:>13}" for era in eras))
    missing = []
    for dataset_name, (files, eras_found) in inputs.items():
        counts = [eras_found.get(era, 0) for era in eras]
        print(
            f"{dataset_name:<{name_width}} "
            + " ".join(f"{count:>13}" for count in counts)
        )
        if len(files) == 0:
            missing.append(dataset_name)
    if len(missing) > 0:
        print(f"WARNING: no input files in any era for {len(missing)} datasets:")
        for dataset_name in missing:
            print(f"  {dataset_name}")


def process_dataset(
    files, dataset_name, class_value, X_mass, config_dict, output_folder
):
    treeName = "Events"
    nParity = config_dict["nParity"]
    rdf = ROOT.RDataFrame(treeName, files)

    # Book every count, sum and snapshot first so they all run in one event loop
    total = rdf.Count()
    rdf = rdf.Filter(config_dict["iterate_cut"])

    snapshot_opts = ROOT.RDF.RSnapshotOptions()
    snapshot_opts.fLazy = True

    results = []
    for parity_scan in range(nParity):
        parity_cut_formatted = config_dict["parity_cut"].format(
            nParity=nParity, parity_scan=parity_scan
        )
        output_nParity = os.path.join(output_folder, f"nParity{parity_scan}_Merged")
        output_file = os.path.join(output_nParity, f"{dataset_name}_merge.root")

        rdf_tmp = rdf.Filter(parity_cut_formatted)
        cut = rdf_tmp.Count()
        weighted_cut = rdf_tmp.Sum("weight_Central")

        if config_dict.get("extra_vars", False):
            rdf_tmp, cols_to_save = add_extra_vars(rdf_tmp, class_value, X_mass)
            snapshot = rdf_tmp.Snapshot(
                treeName, output_file, cols_to_save, snapshot_opts
            )
        else:
            rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
            rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
            snapshot = rdf_tmp.Snapshot(treeName, output_file, "", snapshot_opts)
        results.append((cut, weighted_cut, snapshot))

    total = total.GetValue()
    stats = []
    for cut, weighted_cut, snapshot in results:
        snapshot.GetValue()
        stats.append(
            {
                "total": total,
                "total_cut": cut.GetValue(),
                "total_cut_weighted": weighted_cut.GetValue(),
            }
        )
    return stats


def measure_cut_datasets(config_dict, output_folder, inputs):
    nParity = config_dict["nParity"]
    for parity_scan in range(nParity):
        os.makedirs(
            os.path.join(output_folder, f"nParity{parity_scan}_Merged"), exist_ok=True
        )

    process_dict = {f"nParity_{parity_scan}": {} for parity_scan in range(nParity)}

    for process_name, key, dataset_name, class_value, X_mass in tqdm(
        get_dataset_list(config_dict)
    ):
        files, eras_found = inputs[dataset_name]
        if len(files) == 0:
            print(f"WARNING: skipping {dataset_name}, no input files")
            stats = [{"total": 0, "total_cut": 0, "total_cut_weighted": 0.0}] * nParity
        else:
            stats = process_dataset(
                files, dataset_name, class_value, X_mass, config_dict, output_folder
            )

        for parity_scan in range(nParity):
            process_dict[f"nParity_{parity_scan}"].setdefault(process_name, {})[key] = {
                **stats[parity_scan],
                "eras": eras_found,
            }

    for parity_scan in range(nParity):
        out_yaml = f"dataset_distribution_parity{parity_scan}.yaml"
        with open(os.path.join(output_folder, out_yaml), "w") as outfile:
            yaml.dump(process_dict[f"nParity_{parity_scan}"], outfile)


def hadd_files(config_dict, output_folder):
    for nParity in range(config_dict["nParity"]):
        # hadd the files together to make a final merged.root
        hadd_out = os.path.join(output_folder, f"nParity{nParity}_Merged.root")
        hadd_in = os.path.join(output_folder, f"nParity{nParity}_Merged/*.root")
        os.system(f"hadd {hadd_out} {hadd_in}")


def add_weight_file(output_folder, mass=None, keep_backgrounds=None, suffix=""):
    # keep_backgrounds: optional list of background class_value ints to keep
    # (e.g. [1] for TT-only, [2] for DY-only). Signal (class_value 0) is
    # always kept. Any background class not in this list has its
    # class_weight zeroed out, so it drops out of the training.
    inNames = [
        os.path.join(output_folder, x)
        for x in os.listdir(output_folder)
        if x.endswith(".root")
    ]
    for inName in inNames:
        if "weight" in inName:
            continue
        print(f"On file {inName}")
        in_file = uproot.open(inName)
        outName = f"{inName[:-5]}_weight.root"
        if mass != None:
            outName = f"{inName[:-5]}_weight_m{mass}.root"
        if suffix:
            outName = f"{outName[:-5]}{suffix}.root"
        out_file = uproot.recreate(outName)

        tree = in_file["Events"]
        branches_to_load = [
            "class_value",
            "X_mass",
            "weight_Central",
        ]
        branches = tree.arrays(branches_to_load)

        X_mass = branches["X_mass"]
        class_targets = branches["class_value"]
        class_weight = branches["weight_Central"]

        # Set all signals to target 0
        class_targets = np.where(class_targets <= 0, 0, class_targets)

        # Set to binary for now actually
        class_targets_binary = np.where(class_targets <= 0, 0, 1)

        # Set any negative weight events to 0
        class_weight = np.where(class_weight <= 0, 0.0, class_weight)
        # jk

        # Zero out any background class not in keep_backgrounds (signal,
        # class_value 0, is never dropped here)
        if keep_backgrounds is not None:
            drop_mask = (class_targets != 0) & ~np.isin(
                class_targets, keep_backgrounds
            )
            class_weight = np.where(drop_mask, 0.0, class_weight)

        # Clip weights to be within +- 3 std of mean
        mean_weight = np.mean(np.abs(class_weight))
        std = np.std(np.abs(class_weight))
        print(f"Normalizing from {mean_weight} +- {std}")
        class_weight = np.clip(
            class_weight, -(mean_weight + (3 * std)), (mean_weight + (3 * std))
        )

        # Set specific masses if you want
        # class_weight = np.where((class_targets == 0) & ( (X_mass < 600) | (X_mass > 1000) ), 0.0, class_weight)
        if mass != None:
            class_weight = np.where(
                (class_targets == 0) & ((X_mass != mass)), 0.0, class_weight
            )

        # Total_Signal == Total_Background
        # Scale total signal up to total background
        total_signal = np.sum(np.where(class_targets == 0, class_weight, 0.0))
        total_background = np.sum(np.where(class_targets != 0, class_weight, 0.0))

        print(f"Total signal: {total_signal}")
        print(f"Total background: {total_background}")
        norm_factor = total_background / total_signal
        class_weight = np.where(
            class_targets != 0, class_weight, class_weight * norm_factor
        )

        print(f"After reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, class_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, class_weight, 0.0))}"
        )

        # Total_Background1 == Total_Background2 == Total_Background3
        # Scale each background to total, then reduce all to total
        ### Do not scale backgrounds to each other in binary classifier ###
        total_background = np.sum(np.where(class_targets != 0, class_weight, 0.0))
        multiclass_weight = np.copy(class_weight)
        for class_value in np.unique(class_targets):
            if class_value == 0:
                continue  # Don't do anything with signal here
            this_total = np.sum(
                np.where(class_targets == class_value, class_weight, 0.0)
            )
            if this_total == 0:
                continue  # Class fully dropped (e.g. via keep_backgrounds)
            rescale_factor = total_background / this_total
            multiclass_weight = np.where(
                class_targets == class_value,
                multiclass_weight * rescale_factor,
                multiclass_weight,
            )
        # current_total = np.sum(np.where(class_targets != 0, multiclass_weight, 0.0))
        # rescale_factor = total_background / current_total
        # multiclass_weight = np.where(
        #     class_targets != 0, multiclass_weight*rescale_factor, multiclass_weight
        # )

        # Scale background to nMasses being used
        # mass_cut = (class_targets == 0) * (class_weight > 0.0)
        # nMasses = len(np.unique(X_mass[mass_cut]))
        # print(f"We have {len(np.unique(X_mass[mass_cut]))} unique masses {np.unique(X_mass[mass_cut])}")
        # rescale_factor = nMasses
        # class_weight = np.where(
        #     class_targets != 0, class_weight*rescale_factor, class_weight
        # )

        print(f"Final reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, class_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, class_weight, 0.0))}"
        )
        print(f"And multiclass reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, multiclass_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, multiclass_weight, 0.0))}"
        )

        counts, bin_edges = np.histogram(
            branches["class_value"], bins=15, range=(-5, 10), weights=class_weight
        )
        weighted_histogram = (counts, bin_edges)

        counts_multiclass, bin_edges_multiclass = np.histogram(
            branches["class_value"], bins=15, range=(-5, 10), weights=multiclass_weight
        )
        weighted_histogram_multiclass = (counts_multiclass, bin_edges_multiclass)

        out_dict = {
            "weight_tree": {
                "class_weight": class_weight,
                "class_target": class_targets,
                "multiclass_weight": multiclass_weight,
                "class_targets_binary": class_targets_binary,
            },
            "weighted_class_targets": weighted_histogram,
            "weighted_class_targets_multiclass": weighted_histogram_multiclass,
        }

        print("Finished with dict")
        print(out_dict)

        for key, value in out_dict.items():
            if isinstance(value, tuple):
                val_arr = (
                    value[0].to_numpy() if hasattr(value[0], "to_numpy") else value[0]
                )
                edge_arr = (
                    value[1].to_numpy() if hasattr(value[1], "to_numpy") else value[1]
                )
                out_file[key] = (val_arr, edge_arr)
            else:
                # Explicit TTree: newer uproot writes a plain dict as an RNTuple,
                # which the PyTorch loader cannot read into pandas
                out_file.mktree(
                    key, {name: np.asarray(arr) for name, arr in value.items()}
                )

        out_file.close()


def input_feature_plots(config_dict, output_folder):
    inNames = [
        os.path.join(output_folder, x)
        for x in os.listdir(output_folder)
        if x.endswith(".root")
    ]
    # color_map = plt.get_cmap("tab10").colors[:10]
    color_map = plt.get_cmap("tab20").colors

    input_features = set(
        [
            "nExtraLeps",
            "nExtraTau",
            "lep1_pt",
            "lep2_pt",
            "PuppiMET_pt",
            "HT",
            "MT",
            "MT2_ll",
            "MT2_bb",
            "MT2_blbl1",
            "MT2_blbl2",
            "total_MT",
            "lep1_MT",
            "lep2_MT",
            "ll_mass",
            "ll_pt",
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
            "bb_pt",
            "bb_mass",
            "llmet_mass",
            "bbllmet_mass",
            "bjet1_pt",
            "bjet1_mass",
            "bjet2_pt",
            "bjet2_mass",
            # "m_b1l1", "m_b1l2", "m_b2l1", "m_b2l2",
            # "dR_b1l1", "dR_b1l2", "dR_b2l1", "dR_b2l2",
            # other_jet1/2 do not exist in v2605a (wjet1/2 are SL-only)
            # "other_jet1_pt",
            # "other_jet1_mass",
            # "other_jet2_pt",
            # "other_jet2_mass",
            "fatbjet_pt",
            "fatbjet_mass_PNetCorr",
            "fatbjet_particleNetWithMass_HbbvsQCD",
            # v2605a renames: old names left commented for reference
            # "dR_dilep",
            "ll_dR",
            # "dR_dibjet",
            "bb_dR",
            # "dR_dilep_dibjet",
            "ll_bb_dR",
            # "dPhi_MET_dilep",
            "met_ll_dphi",
            # "dPhi_MET_dibjet",
            "met_bb_dphi",
            "DeepHME_mass",
            "dR_b1leps",
            "dR_b2leps",
            "m_b1leps",
            "m_b2leps",
            "lep1_E",
            "lep1_px",
            "lep1_py",
            "lep1_pz",
            "lep2_E",
            "lep2_px",
            "lep2_py",
            "lep2_pz",
            "bjet1_E",
            "bjet1_px",
            "bjet1_py",
            "bjet1_pz",
            "bjet2_E",
            "bjet2_px",
            "bjet2_py",
            "bjet2_pz",
            # jet3/jet4 were built from other_jets, no longer in the dataset
            # "jet3_E",
            # "jet3_px",
            # "jet3_py",
            # "jet3_pz",
            # "jet4_E",
            # "jet4_px",
            # "jet4_py",
            # "jet4_pz",
            "fatjet_E",
            "fatjet_px",
            "fatjet_py",
            "fatjet_pz",
            "met_E",
            "met_px",
            "met_py",
            "met_pz",
            "bjet1_btagPNetB",
            "bjet2_btagPNetB",
            "fatbjet_tau1",
            "fatbjet_tau2",
            "fatbjet_tau3",
            "fatbjet_tau4",
            # "CosTheta_bb",
            "bb_CosTheta",
            # dijet variables use the SL-only wjet pair in v2605a
            # "dR_dilep_dijet",
            # "dPhi_jet1_jet2",
            "fatbjet_msoftdrop",
            # "dPhi_lep1_lep2",
            "ll_dphi",
        ]
    )
    base_branches = set(["class_value", "X_mass", "weight_Central"])

    class_names = {
        process_dict["class_value"]: process_name
        for process_type in ["signal", "background"]
        for process_name, process_dict in config_dict[process_type].items()
    }

    for inName in inNames:
        if "weight" in inName:
            continue
        print(f"On file {inName} for plots")
        in_file = uproot.open(inName)

        subfolder_name = f"{inName[:-5]}_input_features"
        os.makedirs(subfolder_name, exist_ok=True)

        tree = in_file["Events"]

        missing_features = input_features - set(tree.keys())
        if len(missing_features) > 0:
            print(f"Not plotting features missing from the tree: {missing_features}")
        plot_features = input_features - missing_features
        branches = tree.arrays(list(plot_features | base_branches))

        X_mass = branches["X_mass"]
        class_targets = branches["class_value"]
        class_weight = branches["weight_Central"]

        for inp_feature in plot_features:
            color_map_idx = 0
            # Make a plot of input features with different colors for each class_target
            for class_value in np.unique(class_targets):
                # print(f"Plotting color {color_map_idx} for class {class_value} on map {color_map}")
                feature_values = branches[inp_feature][class_targets == class_value]
                weights = class_weight[class_targets == class_value]
                mass = X_mass[class_targets == class_value]
                feature_quants = np.quantile(feature_values, [0.01, 0.99])
                mask = (feature_values >= feature_quants[0]) & (
                    feature_values <= feature_quants[1]
                )
                class_plot_name = class_names[class_value]
                if class_value == 0:
                    for x_mass in [300, 600, 800]:
                        sig_mask = (mask) & (mass == x_mass)
                        plt.hist(
                            feature_values[sig_mask],
                            bins=50,
                            weights=weights[sig_mask],
                            alpha=0.5,
                            label=f"{class_plot_name} m{x_mass}",
                            # Normalize to 1 for better comparison of shapes
                            density=True,
                            histtype="step",
                            linewidth="1.5",
                            color=color_map[color_map_idx],
                        )
                        color_map_idx += 1
                else:
                    plt.hist(
                        feature_values[mask],
                        bins=50,
                        weights=weights[mask],
                        alpha=0.5,
                        label=f"{class_plot_name}",
                        # Normalize to 1 for better comparison of shapes
                        density=True,
                        histtype="step",
                        linewidth="1.5",
                        color=color_map[color_map_idx],
                    )
                    color_map_idx += 1
            plt.xlabel(inp_feature)
            plt.ylabel("Weighted Events")
            plt.title(f"Distribution of {inp_feature}")
            plt.legend()
            plt.savefig(os.path.join(subfolder_name, f"{inp_feature}_distribution.png"))
            plt.clf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        default="default_dataset.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        type=str,
        default="/eos/user/d/daebi/HH_bbWW/DNNDatasets",
        help="Output folder to store dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the input files found per dataset and era",
    )

    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    storage_folders = get_storage_folders(config_dict)
    print("Collecting input files")
    dataset_names = [dataset[2] for dataset in get_dataset_list(config_dict)]
    # Listing remote eras is one xrdfs call per dataset and era, so run them in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        inputs = dict(
            zip(
                dataset_names,
                executor.map(
                    lambda dataset_name: collect_inputs(storage_folders, dataset_name),
                    dataset_names,
                ),
            )
        )
    print_input_summary(storage_folders, inputs)
    if args.dry_run:
        exit(0)

    output_base = args.output_folder
    output_folder = os.path.join(output_base, f"Dataset")
    if os.path.exists(output_folder):
        print(f"Output folder {output_folder} exists!!!")
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {config_file} {output_folder}/.")

    measure_cut_datasets(config_dict, output_folder, inputs)
    hadd_files(config_dict, output_folder)

    # all_masses: nParity{k}_Merged_weight.root (PyTorch pDNN)
    # per_mass_signal: nParity{k}_Merged_weight_m{mass}.root for each mass of that signal (TF)
    # The PyTorch training computes its weights from weight_Central, so no weight file is
    # written unless the config asks for one (TensorFlow training, legacy PyTorch configs)
    weight_files = config_dict.get("weight_files") or {}
    if weight_files.get("all_masses", False):
        add_weight_file(output_folder)
    per_mass_signal = weight_files.get("per_mass_signal")
    if per_mass_signal is not None:
        for mass in config_dict["signal"][per_mass_signal]["mass_points"]:
            print(f"Starting mass {mass}")
            add_weight_file(output_folder, mass=mass)
    input_feature_plots(config_dict, output_folder)
