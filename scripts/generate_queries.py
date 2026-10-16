"""
generate_queries.py

CLI entry point to generate training queries via an OpenAI model. Real logic lives in
src/data/query_generation.py; this just wires up config + the API key + CLI flags.

Reads the API key from the OPENAI_API_KEY environment variable, auto-loaded from a
.env file at the repo root (via python-dotenv) if present. Never put the key in
config.yaml or commit it to the repo — .env is gitignored.

Example usage:
    # .env at repo root containing: OPENAI_API_KEY=sk-...
    python scripts/generate_queries.py --max-studies 20   # smoke test, check quality/cost
    python scripts/generate_queries.py                    # full run over the corpus, resumable
    python scripts/generate_queries.py --fresh             # wipe queries.jsonl and start over
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from src.utils import load_config
from src.data.query_generation import generate_all

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main():
    parser = argparse.ArgumentParser(description="Generate search queries for each study via an LLM.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-studies", type=int, default=None, help="Cap how many NEW studies to process this run (e.g. for a smoke test).")
    parser.add_argument("--fresh", action="store_true", help="Delete existing output file and start a new run instead of resuming.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set the OPENAI_API_KEY environment variable before running this script.")

    config = load_config(args.config)
    q_config = config["queries"]
    corpus_path = config["data"]["corpus_file"]
    output_path = q_config["output_path"]

    if args.fresh:
        Path(output_path).unlink(missing_ok=True)

    max_studies = args.max_studies if args.max_studies is not None else q_config.get("max_studies")

    generate_all(
        api_key=api_key,
        corpus_path=corpus_path,
        output_path=output_path,
        model=q_config["model"],
        temperature=q_config["temperature"],
        max_input_chars=q_config["max_input_chars"],
        max_studies=max_studies,
        log_every=q_config["log_every"],
        request_timeout=q_config["request_timeout"],
        max_retries=q_config["max_retries"],
        max_workers=q_config["max_workers"],
    )


if __name__ == "__main__":
    main()
