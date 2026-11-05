"""Profiling of the /predict inference path (see monitoring/PERF_STUDY.md).

Two levels of analysis, based on the real monitoring data (first request
~1241 ms in tracking/api_requests.jsonl, then ~10-15 ms steady state):

1. Model path phases, timed separately on real application rows:
   input frame build -> preprocessor.transform -> predict_proba -> predict
   (the current endpoint calls predict AND predict_proba: two passes).
2. Full request level via the in-process API (TestClient): total request
   latency vs the reported processing_time_ms. The difference is everything
   around the model: Pydantic validation, monitoring buffer, request log,
   and above all the synchronous MLflow logging (one sqlite run per
   prediction).

cProfile is run on both levels to rank the bottlenecks.

Usage:
    uv run python monitoring/profile_inference.py --n 200 --out baseline
    uv run python monitoring/profile_inference.py --n 200 --out optimized
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.generate_traffic import build_payloads  # noqa: E402

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "api/models/credit_scoring_model.pkl")
DEFAULT_SCHEMA = os.path.join("api", "schema", "raw_columns.json")
DEFAULT_SOURCE = os.path.join("data", "raw", "application_train.csv")
DEFAULT_OUT_DIR = os.path.join("monitoring", "output")


def load_fields(schema_path: str) -> List[Dict[str, Any]]:
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)["fields"]


def to_raw_frame(payload: Dict[str, Any], fields: List[Dict[str, Any]]) -> pd.DataFrame:
    """Impute like the API does (training median/mode) and build the frame."""
    filled = {}
    for f in fields:
        v = payload.get(f["name"])
        filled[f["name"]] = f["default"] if v is None else v
    return pd.DataFrame([filled])


def stats_ms(timings_ms: List[float]) -> Dict[str, float]:
    s = sorted(timings_ms)

    def pct(p):
        k = (len(s) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {
        "n": len(s),
        "mean_ms": round(statistics.mean(s), 3),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "max_ms": round(s[-1], 3),
    }


def profile_model_path(model, preprocessor, payloads, fields, onnx_probability=None):
    """Phase timings on the model path + cProfile ranking.

    Measures per row: the frame build, the preprocessor transform, then
    three inference variants:
    - lightgbm_double (baseline code: predict_proba + predict),
    - lightgbm_single (optimized code: predict_proba only, class derived),
    - onnx_single (optimized engine, when an ONNX session is provided).
    """
    phases = {"build_frame": [], "preprocessor": [],
              "lightgbm_double": [], "lightgbm_single": []}
    if onnx_probability is not None:
        phases["onnx_single"] = []
    cold_ms = None
    for i, payload in enumerate(payloads):
        t0 = time.perf_counter()
        raw_df = to_raw_frame(payload, fields)
        t1 = time.perf_counter()
        features = preprocessor.transform(raw_df)
        t2 = time.perf_counter()
        model.predict_proba(features)
        model.predict(features)
        t3 = time.perf_counter()
        model.predict_proba(features)
        t4 = time.perf_counter()
        if i == 0:
            cold_ms = (t4 - t0) * 1000
        phases["build_frame"].append((t1 - t0) * 1000)
        phases["preprocessor"].append((t2 - t1) * 1000)
        phases["lightgbm_double"].append((t3 - t2) * 1000)
        phases["lightgbm_single"].append((t4 - t3) * 1000)
        if onnx_probability is not None:
            t5 = time.perf_counter()
            onnx_probability(features)
            phases["onnx_single"].append((time.perf_counter() - t5) * 1000)

    phase_stats = {name: stats_ms(v) for name, v in phases.items()}

    # cProfile over the optimized model path (frame -> preprocessor -> inference)
    def _loop():
        for payload in payloads[:100]:
            features = preprocessor.transform(to_raw_frame(payload, fields))
            model.predict_proba(features)

    prof = cProfile.Profile()
    prof.enable()
    _loop()
    prof.disable()
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(18)
    top = stream.getvalue()

    return {
        "cold_first_call_ms": round(cold_ms, 3) if cold_ms is not None else 0.0,
        "phases": phase_stats,
        "cprofile_top": top,
    }


def profile_request_level(payloads: List[Dict[str, Any]], n_profile: int = 30):
    """End-to-end requests via the in-process API (TestClient).

    Compares the total request latency with the processing_time_ms reported
    by the endpoint: the gap is the serving overhead (validation, logging).
    """
    from fastapi.testclient import TestClient
    from api.main import app

    durations_ms: List[float] = []
    processing_ms: List[float] = []
    first_request_ms = None

    with TestClient(app) as client:  # runs the startup event (model load)
        for i, payload in enumerate(payloads):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=payload)
            t1 = time.perf_counter()
            resp.raise_for_status()
            durations_ms.append((t1 - t0) * 1000)
            processing_ms.append(resp.json()["processing_time_ms"])
            if i == 0:
                first_request_ms = (t1 - t0) * 1000

        def _loop():
            for payload in payloads[:n_profile]:
                client.post("/predict", json=payload)

        prof = cProfile.Profile()
        prof.enable()
        _loop()
        prof.disable()
        stream = io.StringIO()
        pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(20)
        top = stream.getvalue()

    return {
        "first_request_ms": round(first_request_ms, 3) if first_request_ms is not None else 0.0,
        "request_total": stats_ms(durations_ms[1:]),  # exclude the cold one
        "processing_time_ms": stats_ms(processing_ms[1:]),
        "overhead_mean_ms": round(
            statistics.mean(durations_ms[1:]) - statistics.mean(processing_ms[1:]), 3),
        "cprofile_top": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="Number of payloads")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="baseline",
                        help="Written into the output filename (baseline/optimized)")
    args = parser.parse_args()

    import joblib

    print(f"Loading model from {args.model}...")
    artifact = joblib.load(args.model)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    fields = load_fields(args.schema)

    # ONNX engine if available (same loading logic as the API)
    onnx_probability = None
    onnx_path = os.getenv("ONNX_MODEL_PATH", "api/models/credit_scoring_model.onnx")
    if os.path.exists(onnx_path) and artifact.get("feature_names"):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            out_names = [o.name for o in session.get_outputs()]
            output_name = "probabilities" if "probabilities" in out_names else out_names[-1]
            in_name = session.get_inputs()[0].name

            def onnx_probability(X, _s=session, _i=in_name, _o=output_name):
                return float(_s.run([_o], {_i: X.to_numpy().astype("float32")})[0][0][1])
        except ImportError:
            print("onnxruntime not installed: ONNX phase skipped")

    print(f"Building {args.n} payloads from {args.source}...")
    payloads = build_payloads(args.source, args.n, shift=False)

    print("Profiling model path (frame -> preprocessor -> inference variants)...")
    model_profile = profile_model_path(model, preprocessor, payloads, fields,
                                       onnx_probability=onnx_probability)

    print("Profiling full requests (in-process API)...")
    request_profile = profile_request_level(payloads)

    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_payloads": args.n,
        "model_path": model_profile,
        "request_level": request_profile,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"profile_results_{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results written: {out_path}")

    print("\n=== Model path (warm) ===")
    print(f"cold first call (lightgbm): {model_profile['cold_first_call_ms']:.1f} ms")
    for name, st in model_profile["phases"].items():
        print(f"  {name:16s} mean {st['mean_ms']:7.2f} ms | p95 {st['p95_ms']:7.2f} ms")

    rq = request_profile
    print("\n=== Request level (warm, excluding first) ===")
    print(f"  request total   mean {rq['request_total']['mean_ms']:7.2f} ms "
          f"| p95 {rq['request_total']['p95_ms']:7.2f} ms")
    print(f"  processing_time mean {rq['processing_time_ms']['mean_ms']:7.2f} ms "
          f"| p95 {rq['processing_time_ms']['p95_ms']:7.2f} ms")
    print(f"  serving overhead mean {rq['overhead_mean_ms']:.2f} ms "
          "(validation + monitoring buffer + request log + MLflow sqlite)")


if __name__ == "__main__":
    main()
