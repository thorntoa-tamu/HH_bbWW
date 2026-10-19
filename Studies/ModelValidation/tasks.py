import law
import os
import yaml
import shutil
import luigi
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    copy_param,
)
from FLAF.RunKit.run_tools import ps_call
from Studies.DNN.tasks import DNNValidationTask
import re
import contextlib


class DNNLimitTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()
    resolved = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(DNNLimitTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        unique_trainings = {}
        branches = {}
        DNNValidation_map = DNNValidationTask.req(
            self,
            branch=-1,
            branches=(),
            training_configuration_dir=self.training_configuration_dir,
        ).create_branch_map()
        for n_branch, (config, config_name, n_branch) in DNNValidation_map.items():
            training_id = re.search(r"Training\d+", config_name).group(0)
            if training_id not in unique_trainings.keys():
                unique_trainings[training_id] = {
                    "config_names": [],
                    "branches": [],
                }
            unique_trainings[training_id]["config_names"].append(config_name)
            unique_trainings[training_id]["branches"].append(n_branch)

        k = 0
        for training_id in unique_trainings.keys():
            branches[k] = (
                training_id,
                unique_trainings[training_id]["config_names"],
                unique_trainings[training_id]["branches"],
                self.resolved,
            )
            k += 1
        return branches

    def workflow_requires(self):
        branch_set = set()
        for idx, (
            training_id,
            config_names,
            branches,
            resolved,
        ) in self.branch_map.items():
            branch_set.update(branches)
        reqs = {}

        if len(branch_set) > 0:
            reqs["DNNValidation"] = DNNValidationTask.req(
                self,
                training_configuration_dir=self.training_configuration_dir,
                branches=tuple(branch_set),
                customisations=self.customisations,
                max_runtime=DNNValidationTask.max_runtime._default,
                n_cpus=DNNValidationTask.n_cpus._default,
            )

        return reqs

    def requires(self):
        training_id, config_names, branches, resolved = self.branch_data
        return [
            DNNValidationTask.req(
                self,
                training_configuration_dir=self.training_configuration_dir,
                branch=br,
                max_runtime=DNNValidationTask.max_runtime._default,
                branches=(br,),
            )
            for br in branches
        ]

    def output(self):
        training_id, config_names, branches, resolved = self.branch_data
        training_name = training_id
        outFileName = f"limits"
        resolved_or_boosted = "resolved" if int(resolved) else "boosted"
        output_path = os.path.join(
            "DNNLimits",
            self.version,
            self.only_era,
            training_name,
            resolved_or_boosted,
            outFileName,
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        training_id, config_names, branches, resolved = self.branch_data
        training_name = training_id
        dnn_limits = os.path.join(
            self.ana_path(), "Studies", "ModelValidation", "DNN_Limit_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        mass_input_dict = {}

        for inp in self.input():
            mass = ((inp[0].path).split("_m")[1]).split("/")[0]
            if mass not in mass_input_dict.keys():
                mass_input_dict[mass] = []
            mass_input_dict[mass].append(inp)

        print("We have mass-input dict")
        print(mass_input_dict)

        with contextlib.ExitStack() as stack:
            for mass in mass_input_dict.keys():
                inputs = mass_input_dict[mass]
                # inputs = self.input()
                print(f"Starting localize of {len(inputs)} inputs")
                local_inputs = [
                    stack.enter_context(inp[0].localize("r")).path for inp in inputs
                ]

                dnn_limits_cmd = [
                    "python3",
                    "-u",
                    dnn_limits,
                    "--output_folder",
                    training_name,
                    "--resolved",
                    resolved,
                    "--mass",
                    mass,
                    "--validation_paths",
                    *local_inputs,
                ]
                ps_call(dnn_limits_cmd, verbose=1, cwd=job_home)

        for fname in config_names:
            shutil.copy(fname, tmpFolder)

        limit_outputs = self.output()
        with limit_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)


class DNNComparisonTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()
    resolved = luigi.Parameter()

    # fancy_string = "gamma1<5 && gamma2>3"

    def __init__(self, *args, **kwargs):
        super(DNNComparisonTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        unique_trainings = {}
        branches = {}
        DNNLimit_map = DNNLimitTask.req(
            self,
            branch=-1,
            branches=(),
        ).create_branch_map()
        branch_list = []
        for n_branch, (training_id, config_names, _, resolved) in DNNLimit_map.items():
            unique_trainings[training_id] = (n_branch, config_names, resolved)
            branch_list.append(n_branch)

        k = 0
        branches[0] = (unique_trainings, branch_list, self.resolved)
        return branches

    def workflow_requires(self):
        reqs = {}
        reqs["DNNLimit"] = DNNLimitTask.req(
            self,
            branches=tuple(),
            customisations=self.customisations,
            max_runtime=DNNLimitTask.max_runtime._default,
            n_cpus=DNNLimitTask.n_cpus._default,
        )
        return reqs

    def requires(self):
        unique_trainings, branch_list, resolved = self.branch_data
        return [
            DNNLimitTask.req(
                self,
                branch=(br),
                max_runtime=DNNLimitTask.max_runtime._default,
                branches=(br,),
            )
            for br in branch_list
        ]

    def output(self):
        unique_trainings, branch_list, resolved = self.branch_data
        outFileName = f"DNNComparisonPlots"
        resolved_or_boosted = "resolved" if int(resolved) else "boosted"
        output_path = os.path.join(
            "DNNComparison",
            self.version,
            self.only_era,
            resolved_or_boosted,
            outFileName,
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        unique_trainings, branch_list, resolved = self.branch_data

        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"comparison_plots")

        dnn_comparison = os.path.join(
            self.ana_path(), "Studies", "ModelValidation", "DNN_Comparison_Condor.py"
        )

        with contextlib.ExitStack() as stack:
            local_inputs = [
                stack.enter_context(inp[0].localize("r")).path for inp in self.input()
            ]

            dnn_comparison_cmd = [
                "python3",
                "-u",
                dnn_comparison,
                "--output_folder",
                tmpFolder,
                "--resolved",
                resolved,
                "--limit_paths",
                *local_inputs,
            ]
            ps_call(dnn_comparison_cmd, verbose=1, cwd=job_home)

        comparison_outputs = self.output()
        with comparison_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)

            # eval(my_str, env={"gamma1": point["gamma1"]})

            # Here we have localized the limit folders, now we need code to look over the configs to build a dict of training params and their limits

            # Make a dict of just a list of all limits for a given variable point (gamma = 0.5, all limits)
            # ALSO other part that finds and returns training with best limit for fun

            # Then we would make plots of that dict with error bars for up/down to show impact

            # Then copy that final output folder to the transferred law output

            # And given the best training point, do the inverse -- freeze one value and scan the other vars
