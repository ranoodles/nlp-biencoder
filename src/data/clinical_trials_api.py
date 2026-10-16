"""
clinical_trials_api.py

WHAT TO WRITE HERE:
The ONE module in the repo that talks to the network: pull studies from the
ClinicalTrials.gov API v2 and write them to data/raw/studies.jsonl. Called by
scripts/fetch_clinical_trials.py; src/data/preprocessing.py only ever reads the
JSONL this produces, never the API.

Steps to implement:
1. `fetch_page(page_token, page_size) -> dict`: GET request with
   `pageSize` and (if set) `pageToken` params. Response has "studies" +
   "nextPageToken". Add retry/backoff for transient errors (429/5xx).
2. `extract_record(study) -> dict`: flatten the nested API response (see
   study["protocolSection"]["identificationModule"/"descriptionModule"]) into
   {"id": nct_id, "text": briefTitle + briefSummary}, matching what
   preprocessing.py's load_corpus() expects to read.
3. `load_state`/`save_state`: persist {"next_page_token", "num_fetched"} so an
   interrupted run resumes instead of restarting from page 1.
4. `fetch_all(...)`: loop fetch_page -> extract_record -> append to output_path
   (JSONL) -> save_state -> stop on missing nextPageToken or max_studies reached.

Output contract: data/raw/studies.jsonl, one JSON object per line, at minimum
{"id": "<NCT ID>", "text": "<briefTitle + briefSummary or similar>"}.
"""

from pathlib import Path
import requests
import json


def fetch_page(base_url: str, page_token: str) -> dict:
    """Fetch a single page of studies from the ClinicalTrials.gov API.

    TODO: implement with `requests.get(base_url, params=...)`, raise_for_status(),
    return response.json(). Add retry/backoff for transient failures.
    """
    if page_token != "":
      base_url = base_url + "&pageToken=" + page_token
    resp = requests.get(base_url)
    resp.raise_for_status()
    return resp.json()


def extract_record(study: dict) -> dict:
    """Flatten a raw API study object into {"id": ..., "text": ..., ...}.

    TODO: implement per the field paths described above.
    """
    try:
        s_id = study["protocolSection"]["identificationModule"]["nctId"]
    except Exception:
        s_id = ""

    try:
        title = study["protocolSection"]["identificationModule"]["briefTitle"]
    except Exception:
        title = ""

    try:
        ofc_title = study["protocolSection"]["identificationModule"]["officialTitle"]
    except Exception:
        ofc_title = ""

    try:
        summary = study["protocolSection"]["descriptionModule"]["briefSummary"]
    except Exception:
        summary = ""

    try:
        date = study["protocolSection"]["statusModule"]["completionDateStruct"]["date"]
    except Exception:
        date = None

    try:
        eligibility = study["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    except Exception:
        eligibility = ""

    try:
        outcome = study["protocolSection"]["outcomesModule"]["primaryOutcomes"]["description"]
    except Exception:
        outcome = ""

    if date == None:
        return None

    text = f"""
    {title}
    {summary}
    {eligibility}
    {outcome}
    """

    return {
      "id": s_id,
      "date": date,
      "title": ofc_title,
      "text": text,
    } 


def load_state(state_path: str) -> dict:
    """Load pagination state (next_page_token, num_fetched) from disk if present.

    TODO: implement. Return {"next_page_token": None, "num_fetched": 0} if the
    file doesn't exist yet.
    """

    if not Path(state_path).exists():
        return {"next_page_token": None, "num_fetched": 0}
    with open(state_path, "r") as f:
        return json.load(f)


def save_state(state: dict, state_path: str) -> None:
    """Persist pagination state to disk after each page.

    TODO: implement, e.g. json.dump(state, open(state_path, "w")).
    """
    with open(state_path, "w") as f:
        json.dump(state, f)


def make_corpus(
    base_url: str,
    output_path: str,
    state_path: str,
    max_studies: int = None,
) -> None:
    """Page through the API and append flattened records to output_path (JSONL).

    TODO: implement the loop described in step 4 above. Should be resumable:
    calling this again after an interruption should continue from the saved
    next_page_token rather than re-fetching everything.
    """
    page_token = load_state(state_path)["next_page_token"]
    num_fetched = load_state(state_path)["num_fetched"]
    if page_token == "done":
      return
    if not page_token or page_token == "" or page_token == None:
      page_token = ""
    if not num_fetched:
      num_fetched = 0
    while num_fetched < max_studies:
      data = fetch_page(base_url, page_token)
      with open(output_path, "a") as f:
        for raw_study in data["studies"]:
          study = extract_record(raw_study)
          if study:
            f.write(json.dumps(study) + "\n")
            num_fetched += 1
            print(f"\rCurrently fetched {num_fetched} studies out of {max_studies}", end="", flush=True)
            if num_fetched >= max_studies:
              break
      if data["nextPageToken"]:
        page_token = data["nextPageToken"]
      else:
        page_token = "done" # stop fetching
      save_state({"next_page_token": page_token, "num_fetched": num_fetched}, state_path)

