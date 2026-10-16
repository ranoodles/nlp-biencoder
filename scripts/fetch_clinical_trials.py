"""
fetch_clinical_trials.py

WHAT TO WRITE HERE:
CLI entry point to pull studies from the ClinicalTrials.gov API v2. Should be thin —
all real logic lives in src/data/clinical_trials_api.py.

TODO:
1. Parse CLI args if desired (e.g. --config config.yaml, --max-studies, --fresh).
2. `config = load_config(args.config)["api"]` (src/utils.py).
3. If `--fresh`, delete existing output_path/state_path so fetch_all starts clean.
4. `from src.data.clinical_trials_api import fetch_all; fetch_all(...)` with the
   api.* settings from config (base_url, output_path, state_path, page_size),
   overriding max_studies with args.max_studies if provided.

Example usage once implemented:
    python scripts/fetch_clinical_trials.py --max-studies 50   # smoke test
    python scripts/fetch_clinical_trials.py                    # full pull, resumable
    python scripts/fetch_clinical_trials.py --fresh             # start a new snapshot
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import load_config
from src.data.clinical_trials_api import make_corpus

def main():
    config = load_config()["api"]
    make_corpus(
        base_url=config["base_url"],
        output_path=config["output_path"],
        state_path=config["state_path"],
        max_studies=config["max_studies"],
    )

if __name__ == "__main__":
    main()
