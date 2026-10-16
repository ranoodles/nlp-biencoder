"""
run_training.py

WHAT TO WRITE HERE:
CLI entry point to kick off bi-encoder training. Should be thin — all real logic lives
in src/train.py.

TODO:
1. Parse CLI args if desired (e.g. --config config.yaml), or just hardcode the path.
2. `config = load_config(args.config)` (src/utils.py).
3. `set_seed(config["seed"])`.
4. `from src.train import train; train(config)`.

Example usage once implemented:
    python scripts/run_training.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Train the bi-encoder.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # TODO:
    # from src.utils import load_config, set_seed
    # from src.train import train
    # config = load_config(args.config)
    # set_seed(config["seed"])
    # train(config)
    raise NotImplementedError


if __name__ == "__main__":
    main()
