"""Generate production-like traffic against the scoring API.

Sends real Home Credit application rows (sampled from data/raw) to the
/predict endpoint, so the production logging (MLflow input artifacts,
latency metrics, monitoring buffer) gets realistic data to analyze.

Three modes:
- default: in-process API (TestClient) — no server needed, predictions
  are logged to the local MLflow store (tracking/)
- --url http://...: sends the requests to a running API (local uvicorn,
  docker compose, or the deployed Render service)
- --shift: injects a distribution shift in the sent payloads (EXT_SOURCE_2
  scores pushed to 0, incomes multiplied by 4) to demonstrate that the
  drift analysis detects it

Use --save-payloads to export what was sent as JSONL: the cloud-generated
production data can then be analyzed locally (see drift_analysis.py).

Usage:
    uv run python monitoring/generate_traffic.py --n 150
    uv run python monitoring/generate_traffic.py --n 100 --shift \
        --save-payloads monitoring/output/payloads_shifted.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.ml_pipeline import TOP_FORM_FIELDS  # noqa: E402

DEFAULT_SOURCE = os.path.join("data", "raw", "application_train.csv")


def build_payloads(source_csv: str, n: int, shift: bool,
                   seed: int = 42) -> List[Dict[str, Any]]:
    """Sample n application rows and convert them to /predict payloads."""
    df = pd.read_csv(source_csv, usecols=TOP_FORM_FIELDS)
    df = df.sample(n=min(n, len(df)), random_state=seed)

    if shift:
        # Simulated drift: a data provider outage makes EXT_SOURCE_2 crash
        # towards 0, and a marketing campaign attracts high-income clients.
        df["EXT_SOURCE_2"] = df["EXT_SOURCE_2"] * 0.15
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * 4.0

    import numpy as np

    def _clean(v):
        if pd.isna(v):
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    records = df.to_dict(orient="records")
    return [{str(k): _clean(v) for k, v in row.items()}
            for row in records]


def send_inprocess(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the payloads through an in-process API (TestClient).

    Loads the real model and logs each prediction to the local MLflow
    store, exactly like a deployed instance would.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    results = []
    with TestClient(app) as client:
        for payload in payloads:
            response = client.post("/predict", json=payload)
            results.append({"status_code": response.status_code})
    return results


def send_to_url(payloads: List[Dict[str, Any]], url: str) -> List[Dict[str, Any]]:
    """POST the payloads to a running API."""
    import requests

    results: List[Dict[str, Any]] = []
    for payload in payloads:
        try:
            response = requests.post(f"{url.rstrip('/')}/predict", json=payload, timeout=30)
            results.append({"status_code": response.status_code})
        except requests.RequestException as e:
            results.append({"error": str(e)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="CSV to sample application rows from")
    parser.add_argument("--url", default=None,
                        help="Target API URL (default: in-process API, local MLflow)")
    parser.add_argument("--shift", action="store_true",
                        help="Inject a distribution shift (EXT_SOURCE_2, AMT_INCOME_TOTAL)")
    parser.add_argument("--save-payloads", default=None,
                        help="Write the sent payloads to this JSONL file")
    parser.add_argument("--save-only", action="store_true",
                        help="Only write the payloads file, do not send "
                             "(scenario testing without polluting MLflow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label = "shifted" if args.shift else "normal"
    print(f"Building {args.n} {label} payloads from {args.source}...")
    payloads = build_payloads(args.source, args.n, args.shift, seed=args.seed)

    if args.save_payloads:
        os.makedirs(os.path.dirname(args.save_payloads) or ".", exist_ok=True)
        with open(args.save_payloads, "w", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload) + "\n")
        print(f"Payloads saved: {args.save_payloads}")

    if args.save_only:
        print("--save-only: not sending anything.")
        return

    if args.url:
        print(f"Sending to {args.url} ...")
        results = send_to_url(payloads, args.url)
    else:
        print("Sending through the in-process API (local MLflow store)...")
        results = send_inprocess(payloads)

    ok = sum(1 for r in results if r.get("status_code") == 200)
    errors = [r for r in results if r.get("status_code") != 200]
    print(f"Done: {ok}/{len(results)} predictions succeeded")
    if errors:
        print(f"Errors: {errors[:3]}")


if __name__ == "__main__":
    main()
