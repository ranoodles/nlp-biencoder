"""
query_generation.py

Generates training/eval queries for the bi-encoder: for each study in
data/raw/studies.jsonl, calls an OpenAI chat model to produce two natural-language
search queries, then appends {"id": ..., "queries": [q1, q2]} to data/raw/queries.jsonl.

Calls run concurrently (ThreadPoolExecutor, capped by max_workers) since this is one
independent API call per study — a purely I/O-bound wait, so threads parallelize it
well without needing async. Writes are protected by a lock so concurrent workers can
safely append to the same output file.

Resumable like clinical_trials_api.py's make_corpus: on startup, reads whatever's
already in output_path and skips those ids, so an interrupted/killed run can just be
re-launched. Use --fresh (in scripts/generate_queries.py) to wipe and start over.
"""

import json
import time
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, BadRequestError


class _TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() instead of a plain stream, so they
    print cleanly above the progress bars instead of corrupting their rendering."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


# Quiet the underlying HTTP client's per-request logging (httpx/httpcore log every
# "HTTP Request: POST ..." at INFO level by default) — that's the "all the API POST
# requests" spam. Progress is shown via the two tqdm bars in generate_all() instead.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_TqdmLoggingHandler())
logger.propagate = False

# Some models (e.g. reasoning-style models) reject a custom `temperature` value and
# only support the API default. Detected lazily on first failure and then shared
# across all worker threads so we stop sending it for every subsequent call/study.
_temperature_supported = True
_temperature_lock = threading.Lock()


SYSTEM_PROMPT = """You are generating training queries for a clinical-trial semantic retrieval system.

Given a clinical trial, generate TWO realistic natural-language search queries that a researcher, clinician, or patient might use to find this trial.

Requirements:
1. The query must describe the underlying clinical need, condition, intervention, population, or study objective.
2. Do NOT copy the trial title.
3. Do NOT merely paraphrase the title.
4. Use natural search language rather than database keywords.
5. The two queries should be meaningfully different in wording and emphasis.
6. Only use information supported by the trial.
7. Do not mention the trial ID.
8. Do not make up eligibility criteria, outcomes, interventions, or conditions.
9. Each query should usually be around 8-20 words, using natural language rather than formal trial terminology.
10. Avoid being so vague that essentially every trial could match.
11. Generate diverse queries.

Respond with ONLY a json object of the form {"queries": ["query 1", "query 2"]}. No explanation, no markdown, no extra fields."""

USER_PROMPT_TEMPLATE = """Trial title: {title}

Trial description and details:
{text}"""


def iter_studies(corpus_path: str):
    """Yield study dicts ({"id", "title", "text", ...}) from a JSONL corpus file."""
    with open(corpus_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_done_ids(output_path: str) -> set:
    """Read whatever's already in output_path and return the set of study ids
    already processed, so generate_all() can skip them on a resumed run."""
    path = Path(output_path)
    if not path.exists():
        return set()
    done = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_user_prompt(study: dict, max_input_chars: int) -> str:
    """Build the user-turn prompt for a single study, truncating text to control cost."""
    title = study.get("title", "")
    text = (study.get("text") or "").strip()
    if len(text) > max_input_chars:
        text = text[:max_input_chars]
    return USER_PROMPT_TEMPLATE.format(title=title, text=text)


def generate_queries_for_study(
    client: OpenAI,
    study: dict,
    model: str,
    temperature: float,
    max_input_chars: int,
    request_timeout: int,
    max_retries: int,
    on_request_sent=None,
) -> list:
    """Call the LLM for one study, returning a list of exactly 2 query strings.

    Retries with exponential backoff on rate limits / transient API errors / bad
    JSON output. Raises the last error if all retries are exhausted.

    on_request_sent, if given, is called once per actual HTTP attempt (including
    retries) — used by generate_all() to drive the "requests sent" progress bar.
    """
    user_prompt = build_user_prompt(study, max_input_chars)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    global _temperature_supported
    last_error = None
    for attempt in range(1, max_retries + 1):
        kwargs = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "timeout": request_timeout,
        }
        if _temperature_supported:
            kwargs["temperature"] = temperature

        try:
            if on_request_sent is not None:
                on_request_sent()
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            parsed = json.loads(content)
            queries = parsed["queries"]
            if not isinstance(queries, list) or len(queries) != 2:
                raise ValueError(f"Expected exactly 2 queries, got: {queries!r}")
            return [str(q).strip() for q in queries]
        except BadRequestError as e:
            last_error = e
            if _temperature_supported and "temperature" in str(e).lower():
                with _temperature_lock:
                    if _temperature_supported:
                        _temperature_supported = False
                logger.info(
                    "Model %s rejected a custom temperature; retrying without it "
                    "(this only needs to happen once).",
                    model,
                )
                continue  # retry immediately without consuming a backoff sleep
            logger.warning(
                "Bad request calling model for %s (attempt %d/%d): %s",
                study.get("id"), attempt, max_retries, e,
            )
            time.sleep(1)
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_error = e
            sleep_s = min(2 ** attempt, 30)
            logger.warning(
                "API error on %s (attempt %d/%d): %s — retrying in %ds",
                study.get("id"), attempt, max_retries, e, sleep_s,
            )
            time.sleep(sleep_s)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            logger.warning(
                "Bad model output for %s (attempt %d/%d): %s",
                study.get("id"), attempt, max_retries, e,
            )
            time.sleep(1)

    raise RuntimeError(f"Failed to generate queries for {study.get('id')} after {max_retries} attempts") from last_error


def generate_all(
    api_key: str,
    corpus_path: str,
    output_path: str,
    model: str = "gpt-6-luna",
    temperature: float = 0.7,
    max_input_chars: int = 1600,
    max_studies: int = None,
    log_every: int = 50,
    request_timeout: int = 30,
    max_retries: int = 5,
    max_workers: int = 8,
) -> None:
    """Generate queries for every study in corpus_path not already in output_path.

    Runs up to max_workers studies concurrently (separate, independent API calls),
    appending each result to output_path as soon as it finishes — so progress survives
    interruptions and workers finishing out of order is fine. Call this again to
    resume; already-done ids are skipped.

    Shows two live progress bars instead of per-request log lines: one tracking raw
    API requests sent (ticks on every attempt, including retries), one tracking
    studies that got a valid response back. log_every is unused now (kept for
    backwards-compatible call signatures) — the bars replace periodic log prints.
    """
    client = OpenAI(api_key=api_key)
    done_ids = load_done_ids(output_path)
    if done_ids:
        logger.info("Resuming: %d studies already have queries, skipping them.", len(done_ids))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Decide up front which studies still need queries, capped at max_studies.
    pending = []
    for study in iter_studies(corpus_path):
        study_id = study.get("id")
        if not study_id or study_id in done_ids:
            continue
        pending.append(study)
        if max_studies is not None and len(pending) >= max_studies:
            break

    if not pending:
        logger.info("Nothing to do — all studies already have queries.")
        return

    logger.info("Generating queries for %d studies using %d concurrent workers.", len(pending), max_workers)

    write_lock = threading.Lock()
    counts_lock = threading.Lock()
    sent_lock = threading.Lock()
    num_generated = 0
    num_failed = 0

    sent_bar = tqdm(total=len(pending), desc="Requests sent      ", position=0, leave=True, dynamic_ncols=True)
    success_bar = tqdm(total=len(pending), desc="Successful responses", position=1, leave=True, dynamic_ncols=True)

    def on_request_sent() -> None:
        with sent_lock:
            sent_bar.update(1)

    try:
        with open(output_path, "a") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_study = {
                executor.submit(
                    generate_queries_for_study,
                    client, study, model, temperature, max_input_chars, request_timeout, max_retries,
                    on_request_sent,
                ): study
                for study in pending
            }

            for future in as_completed(future_to_study):
                study = future_to_study[future]
                study_id = study["id"]

                try:
                    queries = future.result()
                except RuntimeError as e:
                    logger.error("Giving up on %s: %s", study_id, e)
                    with counts_lock:
                        num_failed += 1
                    continue

                with write_lock:
                    out_f.write(json.dumps({"id": study_id, "queries": queries}) + "\n")
                    out_f.flush()

                with counts_lock:
                    num_generated += 1
                success_bar.update(1)
    finally:
        sent_bar.close()
        success_bar.close()

    logger.info(
        "Done. Generated=%d, skipped(already done)=%d, failed=%d.",
        num_generated, len(done_ids), num_failed,
    )
