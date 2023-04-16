import argparse
from typing import Dict

import yaml


def config_from_cli_or_yaml() -> Dict:
    """Returns the training configuration for the model.

    If program is called with the command:

        python train_model.py --config config.yaml

    Use the configuration in the config.yaml file.

    If the program is called with the command:

        python train_model.py
            --epochs 100
            --split 0.2
            --batch-size 256
            --dataset-path trainingImages/
            --target-shape 64 64 3
            --model-path model.tf
            --dataset-save-path "dataset.pickle"

    Use the configuration provided in the command line arguments.

    Returns:
        A dictionary containing the configuration for the model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--split", type=float, required=False)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--dataset-path", type=str, required=False)
    parser.add_argument("--target-shape", type=int, nargs=3, required=False)
    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--dataset-save-path", type=str, required=False)
    args = parser.parse_args()

    # If a config file is provided, load it
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        return config

    # Otherwise, return the command line arguments
    return {
        "epochs": args.epochs,
        "split": args.split,
        "batch_size": args.batch_size,
        "dataset_path": args.dataset_path,
        "target_shape": tuple(args.target_shape),
        "model_path": args.model_path,
        "dataset_save_path": args.dataset_save_path,
    }
