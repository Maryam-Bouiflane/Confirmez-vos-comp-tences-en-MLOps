"""Convert the LightGBM model to ONNX and benchmark it against the original.

The preprocessor (custom CreditPreprocessor) is kept in scikit-learn: only
the LGBMClassifier inference is exported to ONNX (single-row latency is
what the /predict endpoint pays on every call). The script:

1. converts the model (zipmap disabled: raw probability tensor, faster
   and easier to read than the default key-value output),
2. saves api/models/credit_scoring_model.onnx (committed: deployed with
   the Docker image via the existing CI/CD pipeline),
3. checks non-regression on N real rows: max probability deviation and
   decision flips (at the sklearn 0.5 cut AND at the business threshold),
4. benchmarks single-row predict_proba: LightGBM vs ONNX Runtime
   (mean / p50 / p95, after warm-up).

Usage:
    uv run --extra onnx python monitoring/optimize_model.py
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.generate_traffic import build_payloads  # noqa: E402

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "api/models/credit_scoring_model.pkl")
DEFAULT_SCHEMA = os.path.join("api", "schema", "raw_columns.json")
DEFAULT_SOURCE = os.path.join("data", "raw", "application_train.csv")
DEFAULT_ONNX_PATH = os.path.join("api", "models", "credit_scoring_model.onnx")
DEFAULT_OUT_DIR = os.path.join("monitoring", "output")


def to_raw_frame(payload: Dict[str, Any], fields: List[Dict[str, Any]]) -> pd.DataFrame:
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

    return {"n": len(s), "mean_ms": round(statistics.mean(s), 3),
            "p50_ms": round(pct(50), 3), "p95_ms": round(pct(95), 3),
            "max_ms": round(s[-1], 3)}


def convert_to_onnx(model: Any, n_features: int, onnx_path: str) -> Any:
    """Convert LGBMClassifier -> ONNX (zipmap=False) and save."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        zipmap=False,
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return onnx_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--onnx-path", default=DEFAULT_ONNX_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=300, help="Rows for parity + benchmark")
    args = parser.parse_args()

    import joblib
    import onnxruntime as ort

    artifact = joblib.load(args.model)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    threshold = float(artifact.get("best_threshold") or 0.5)
    with open(args.schema, encoding="utf-8") as f:
        fields = json.load(f)["fields"]

    print(f"Building {args.n} feature rows (preprocessor applied)...")
    payloads = build_payloads(args.source, args.n, shift=False)
    frames = [to_raw_frame(p, fields) for p in payloads]
    X_list: List[pd.DataFrame] = [preprocessor.transform(fr) for fr in frames]
    n_features = X_list[0].shape[1]
    print(f"Feature matrix: {n_features} columns")

    print("Converting LightGBM -> ONNX...")
    convert_to_onnx(model, n_features, args.onnx_path)
    size_kb = os.path.getsize(args.onnx_path) / 1024
    print(f"Saved {args.onnx_path} ({size_kb:.0f} KiB)")

    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # Two outputs: 'label' (predicted class) and 'probabilities' (N, 2)
    out_names = [o.name for o in session.get_outputs()]
    output_name = "probabilities" if "probabilities" in out_names else out_names[-1]

    # --- Non-regression check on all N rows ------------------------------
    max_diff = 0.0
    flips_05 = 0
    flips_threshold = 0
    for X in X_list:
        p_skl = float(model.predict_proba(X)[0][1])
        p_onnx = float(session.run([output_name], {input_name: X.to_numpy().astype(np.float32)})[0][0][1])
        max_diff = max(max_diff, abs(p_skl - p_onnx))
        if (p_skl > 0.5) != (p_onnx > 0.5):
            flips_05 += 1
        if (p_skl > threshold) != (p_onnx > threshold):
            flips_threshold += 1
    print(f"Parity on {len(X_list)} rows: max |delta proba| = {max_diff:.2e}, "
          f"decision flips (0.5): {flips_05}, (threshold {threshold:.3f}): {flips_threshold}")

    # --- Single-row benchmark (warm) --------------------------------------
    def bench(fn) -> List[float]:
        timings: List[float] = []
        for _ in range(10):  # warm-up
            fn(X_list[0])
        for X in X_list:
            t0 = time.perf_counter()
            fn(X)
            timings.append((time.perf_counter() - t0) * 1000)
        return timings

    skl_timings = bench(lambda X: model.predict_proba(X))
    onnx_timings = bench(lambda X: session.run(
        [output_name], {input_name: X.to_numpy().astype(np.float32)}))

    skl_stats = stats_ms(skl_timings)
    onnx_stats = stats_ms(onnx_timings)

    # --- Batch benchmark (context: future batch endpoints) ----------------
    X_batch = pd.concat(X_list[:32], ignore_index=True).to_numpy().astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(10):
        model.predict_proba(X_batch)
    skl_batch = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        session.run([output_name], {input_name: X_batch})
    onnx_batch = (time.perf_counter() - t0) / 10 * 1000

    speedup = skl_stats["mean_ms"] / onnx_stats["mean_ms"] if onnx_stats["mean_ms"] else 0
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(X_list),
        "n_features": int(n_features),
        "onnx_path": args.onnx_path,
        "onnx_size_kb": round(size_kb, 1),
        "parity": {
            "max_abs_prob_delta": max_diff,
            "decision_flips_at_0_5": flips_05,
            "decision_flips_at_business_threshold": flips_threshold,
            "business_threshold": threshold,
        },
        "single_row_lightgbm": skl_stats,
        "single_row_onnxruntime": onnx_stats,
        "single_row_speedup_x": round(speedup, 2),
        "batch32_lightgbm_ms": round(skl_batch, 3),
        "batch32_onnxruntime_ms": round(onnx_batch, 3),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "optimization_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results written: {out_path}")

    print("\n=== Single-row predict_proba (warm) ===")
    print(f"  LightGBM    mean {skl_stats['mean_ms']:.2f} ms | p50 {skl_stats['p50_ms']:.2f} | p95 {skl_stats['p95_ms']:.2f}")
    print(f"  ONNX Runtime mean {onnx_stats['mean_ms']:.2f} ms | p50 {onnx_stats['p50_ms']:.2f} | p95 {onnx_stats['p95_ms']:.2f}")
    print(f"  speedup x{speedup:.2f}")
    print(f"Batch(32): LightGBM {skl_batch:.2f} ms | ONNX {onnx_batch:.2f} ms")


if __name__ == "__main__":
    main()
