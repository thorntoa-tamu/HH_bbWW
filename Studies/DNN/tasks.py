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


class DNNTrainingTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 48.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 8)

    def __init__(self, *args, **kwargs):
        super(DNNTrainingTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNN_Configurations = [
            os.path.join(self.training_configuration_dir, x)
            for x in os.listdir(self.training_configuration_dir)
            if x.endswith(".yaml")
        ]
        DNN_Configurations.sort()
        # open yaml training_configuration
        for DNN_Configuration in DNN_Configurations:
            with open(DNN_Configuration, "r") as f:
                config = yaml.safe_load(f)
            br_idx = len(branches)
            branches[br_idx] = (config, DNN_Configuration)
        return branches

    def output(self):
        config, config_name = self.branch_data
        training_name = config["training_name"]
        outFolderName = f"{training_name}"
        output_path = os.path.join(
            "DNNTraining", self.version, self.period, training_name, outFolderName
        )
        config_path = os.path.join(
            "DNNTraining",
            self.version,
            self.period,
            training_name,
            os.path.basename(config_name),
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
            self.remote_target(config_path, fs=self.fs_histograms),
        ]

    def run(self):
        config, config_name = self.branch_data
        training_name = config["training_name"]
        dnn_trainer = os.path.join(
            self.ana_path(), "Studies", "DNN", "DNN_Trainer_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        training_file = config["training_file"]
        weight_file = config["weight_file"]
        test_training_file = config["test_training_file"]
        test_weight_file = config["test_weight_file"]

        dnn_trainer_cmd = [
            "python3",
            "-u",
            dnn_trainer,
            "--training_file",
            training_file,
            "--weight_file",
            weight_file,
            "--test_training_file",
            test_training_file,
            "--test_weight_file",
            test_weight_file,
            "--output_folder",
            tmpFolder,
            "--setup-config",
            config_name,
        ]
        ps_call(dnn_trainer_cmd, verbose=1)

        model_output = self.output()[0]
        with model_output.localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)
        config_output = self.output()[1]
        with config_output.localize("w") as tmp_local_file:
            out_local_path = tmp_local_file.path
            shutil.copy(config_name, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)


class DNNValidationTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def __init__(self, *args, **kwargs):
        super(DNNValidationTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNNTraining_map = DNNTrainingTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        k = 0
        for n_branch, (config, config_name) in DNNTraining_map.items():
            branches[k] = (config, config_name, n_branch)
            k += 1
        return branches

    def workflow_requires(self):
        return {"DNNTrainer": DNNTrainingTask.req(self)}

    def requires(self):
        config, config_name, n_branch = self.branch_data
        return DNNTrainingTask.req(
            self,
            branch=n_branch,
            max_runtime=DNNTrainingTask.max_runtime._default,
            branches=(),
        )

    def output(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        outFileName = f"validation"
        output_path = os.path.join(
            "DNNTraining", self.version, self.period, training_name, outFileName
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        dnn_validator = os.path.join(
            self.ana_path(), "Studies", "DNN", "DNN_Validator_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        validation_file = config["validation_file"]
        valitation_weight_file = config["validation_weight_file"]

        # tmp_local = os.path.join(self.input()[0].path, "best.onnx")
        tmp_local = os.path.join(self.input()[0].path, "stage1.onnx")

        with self.remote_target(tmp_local, fs=self.fs_histograms).localize(
            "r"
        ) as model_file, self.input()[1].localize("r") as model_config:
            print(os.listdir())
            dnn_validator_cmd = [
                "python3",
                "-u",
                dnn_validator,
                "--validation_file",
                validation_file,
                "--validation_weight_file",
                valitation_weight_file,
                "--output_folder",
                tmpFolder,
                "--setup-config",
                config_name,
                "--model-name",
                model_file.path,
                "--model-config",
                model_config.path,
            ]
            ps_call(dnn_validator_cmd, verbose=1)

        validation_outputs = self.output()
        with validation_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)


class ABCDValidationTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    """Validate the ABCDisCo (arXiv:2007.14400) assumptions for each training.

    For every training configuration, runs Studies/DNN/abcd_validator.py over
    all epoch_N.onnx checkpoints (plus best.onnx / stage1.onnx) of the
    corresponding DNNTrainingTask output: ABCD closure, control-region
    contamination delta_i, normalized signal contamination r, dCorr(f, mbb),
    plus the paper's r-vs-rejection / closure-vs-rejection / ROC plots and an
    across-epoch scan for choosing the training epoch by ABCD criteria.
    """

    training_configuration_dir = luigi.Parameter()
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 24.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)
    epoch_step = luigi.IntParameter(
        default=1,
        description="validate every Nth epoch checkpoint (last one always kept)",
    )
    mass_points = luigi.Parameter(
        default="",
        description="comma-separated mass points; default: all masses in the "
        "model config that are present in the validation file",
    )
    full_pdfs = luigi.BoolParameter(
        default=False,
        description="write the full per-mass plot PDF for every epoch "
        "checkpoint, not only the final model",
    )
    htcondor_gpus = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="number of GPUs to request for the condor job; the "
        "validator uses the CUDA onnxruntime provider when the worker "
        "environment provides one, otherwise it falls back to CPU",
    )

    def __init__(self, *args, **kwargs):
        super(ABCDValidationTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNNTraining_map = DNNTrainingTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        k = 0
        for n_branch, (config, config_name) in DNNTraining_map.items():
            branches[k] = (config, config_name, n_branch)
            k += 1
        return branches

    def workflow_requires(self):
        return {"DNNTrainer": DNNTrainingTask.req(self)}

    def requires(self):
        config, config_name, n_branch = self.branch_data
        return DNNTrainingTask.req(
            self,
            branch=n_branch,
            max_runtime=DNNTrainingTask.max_runtime._default,
            branches=(),
        )

    def htcondor_job_config(self, config, job_num, branches):
        config = super(ABCDValidationTask, self).htcondor_job_config(
            config, job_num, branches
        )
        if not law.is_no_param(self.htcondor_gpus):
            config.custom_content.append(("request_gpus", self.htcondor_gpus))
        return config

    def output(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        outFileName = f"abcd_validation"
        output_path = os.path.join(
            "DNNTraining", self.version, self.period, training_name, outFileName
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        abcd_validator = os.path.join(
            self.ana_path(), "Studies", "DNN", "ABCD_Validator_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        validation_file = config["validation_file"]
        validation_weight_file = config["validation_weight_file"]
        hme_friend_file = config.get("validation_hme_friend_file", None)

        with self.input()[0].localize("r") as model_folder, self.input()[1].localize(
            "r"
        ) as training_config:
            # the DisCo trainer writes dnn_config.yaml next to the models; it
            # has the resolved feature lists, so prefer it over the training
            # configuration when present
            model_config_path = os.path.join(model_folder.path, "dnn_config.yaml")
            if not os.path.exists(model_config_path):
                model_config_path = training_config.path

            abcd_validator_cmd = [
                "python3",
                "-u",
                abcd_validator,
                "--validation_file",
                validation_file,
                "--validation_weight_file",
                validation_weight_file,
                "--output_folder",
                tmpFolder,
                "--setup-config",
                config_name,
                "--model-folder",
                model_folder.path,
                "--model-config",
                model_config_path,
                "--epoch-step",
                str(self.epoch_step),
            ]
            if hme_friend_file:
                abcd_validator_cmd += ["--hme-friend-file", hme_friend_file]
            if self.mass_points:
                abcd_validator_cmd += ["--mass-points", self.mass_points]
            if self.full_pdfs:
                abcd_validator_cmd += ["--full-pdfs"]
            ps_call(abcd_validator_cmd, verbose=1)

        validation_outputs = self.output()
        with validation_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)


class DNNValidation2Task(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def __init__(self, *args, **kwargs):
        super(DNNValidation2Task, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNNTraining_map = DNNTrainingTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        k = 0
        for n_branch, (config, config_name) in DNNTraining_map.items():
            branches[k] = (config, config_name, n_branch)
            k += 1
        return branches

    def workflow_requires(self):
        return {"DNNTrainer": DNNTrainingTask.req(self)}

    def requires(self):
        config, config_name, n_branch = self.branch_data
        return DNNTrainingTask.req(
            self,
            branch=n_branch,
            max_runtime=DNNTrainingTask.max_runtime._default,
            branches=(),
        )

    def output(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        outFileName = f"validation"
        output_path = os.path.join(
            "DNNTraining", self.version, self.period, training_name, outFileName
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        dnn_validator = os.path.join(
            self.ana_path(), "Studies", "DNN", "DNN_Validator_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        validation_file = config["validation_file"]
        valitation_weight_file = config["validation_weight_file"]

        tmp_local_1 = os.path.join(self.input()[0].path, "stage1.onnx")
        tmp_local_2 = os.path.join(self.input()[0].path, "stage2.onnx")

        with self.remote_target(tmp_local_1, fs=self.fs_histograms).localize(
            "r"
        ) as model_file1, self.remote_target(
            tmp_local_2, fs=self.fs_histograms
        ).localize(
            "r"
        ) as model_file2, self.input()[
            1
        ].localize(
            "r"
        ) as model_config:
            print(os.listdir())
            dnn_validator_cmd = [
                "python3",
                "-u",
                dnn_validator,
                "--validation_file",
                validation_file,
                "--validation_weight_file",
                valitation_weight_file,
                "--output_folder",
                tmpFolder,
                "--setup-config",
                config_name,
                "--model-name",
                model_file1.path,
                "--model-name-stage2",
                model_file2.path,
                "--model-config",
                model_config.path,
            ]
            ps_call(dnn_validator_cmd, verbose=1)

        validation_outputs = self.output()
        with validation_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)
