import argparse
import threading
from FLAF.RunKit.kinit import cond as kInit_cond, update_kinit_thread

import Studies.DNN.abcd_validator as abcd_validator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ABCD(isCo) assumption validation on a training output."
    )
    parser.add_argument(
        "--validation_file", required=True, type=str, help="Validation batch file"
    )
    parser.add_argument(
        "--validation_weight_file", required=True, type=str, help="Weight file"
    )
    parser.add_argument(
        "--output_folder", required=True, type=str, help="Output Folder"
    )
    parser.add_argument(
        "--setup-config", required=False, type=str, help="Setup config for training"
    )
    parser.add_argument(
        "--model-folder",
        required=True,
        type=str,
        help="Training output folder with epoch_N.onnx / best.onnx / stage1.onnx",
    )
    parser.add_argument(
        "--model-config", required=True, type=str, help="Config file for model"
    )
    parser.add_argument(
        "--hme-friend-file", required=False, type=str, help="HME friend file"
    )
    parser.add_argument(
        "--epoch-step",
        required=False,
        type=int,
        default=1,
        help="Validate every Nth epoch checkpoint (last one always kept)",
    )
    parser.add_argument(
        "--mass-points",
        required=False,
        type=str,
        default="",
        help="Comma-separated mass points (default: all in the model config)",
    )
    parser.add_argument(
        "--full-pdfs",
        action="store_true",
        help="Write the full per-mass PDF for every epoch checkpoint",
    )

    args = parser.parse_args()

    validator_argv = [
        "--validation-file",
        args.validation_file,
        "--validation-weight-file",
        args.validation_weight_file,
        "--model-dir",
        args.model_folder,
        "--model-config",
        args.model_config,
        "--output-folder",
        args.output_folder,
        "--epoch-step",
        str(args.epoch_step),
    ]
    if args.hme_friend_file:
        validator_argv += ["--hme-friend-file", args.hme_friend_file]
    if args.mass_points:
        validator_argv += ["--mass-points"] + args.mass_points.split(",")
    if args.full_pdfs:
        validator_argv += ["--full-pdfs"]

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        abcd_validator.main(validator_argv)

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
