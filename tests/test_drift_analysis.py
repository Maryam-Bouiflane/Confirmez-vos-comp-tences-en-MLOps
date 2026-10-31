"""
Tests for the drift analysis tooling (monitoring/drift_analysis.py).

Covers:
- Schema alignment (null imputation with the training median/mode)
- Latency anomaly detection (p95 threshold)
- Error-rate analysis from the API's JSONL request log
- Drift detection on synthetic distributions (Evidently, skipped if not
  installed: the dependency lives in the dev group)
"""

import json

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_analysis import (
    align_to_schema,
    analyze_errors,
    analyze_latency,
    load_requests_log,
    run_drift,
)

FIELDS = [
    {"name": "num_a", "type": "numeric", "default": 0.5},
    {"name": "num_b", "type": "numeric", "default": 10.0},
    {"name": "cat_a", "type": "categorical", "default": "x"},
]


class TestAlignToSchema:
    """Production payloads must be imputed like the API does server-side."""

    def test_imputes_nulls_with_schema_defaults(self):
        df = pd.DataFrame({
            "num_a": [None, 1.0],
            "num_b": [2.0, None],
            "cat_a": [None, "y"],
        })
        out = align_to_schema(df, FIELDS)
        assert out["num_a"].tolist() == [0.5, 1.0]
        assert out["num_b"].tolist() == [2.0, 10.0]
        assert out["cat_a"].tolist() == ["x", "y"]

    def test_missing_column_falls_back_to_default(self):
        """Direct API calls can omit fields entirely."""
        out = align_to_schema(pd.DataFrame({"num_a": [1.0]}), FIELDS)
        assert (out["num_a"] == 1.0).all()
        assert (out["cat_a"] == "x").all()
        assert (out["num_b"] == 10.0).all()

    def test_invalid_numeric_becomes_default(self):
        out = align_to_schema(pd.DataFrame({"num_a": ["oops", 1.0]}), FIELDS)
        assert out["num_a"].tolist() == [0.5, 1.0]


class TestLatencyAnalysis:
    def test_flags_p95_anomaly(self):
        latencies = [10.0] * 90 + [2000.0] * 10
        result = analyze_latency(latencies, p95_alert_ms=500.0)
        assert result["p95_ms"] > 500
        assert result["p95_anomaly"] is True

    def test_normal_latency_no_anomaly(self):
        result = analyze_latency([10.0, 12.0, 11.0, 13.0], p95_alert_ms=500.0)
        assert result["p95_anomaly"] is False
        assert result["avg_ms"] < 20

    def test_empty_latency(self):
        assert analyze_latency([])["n_samples"] == 0
        assert analyze_latency([])["p95_anomaly"] is None


class TestErrorRateAnalysis:
    """Operational analysis of the API's JSONL request log."""

    @staticmethod
    def _event(status_code, path="/predict"):
        return {"timestamp": "2026-10-17T00:00:00+00:00", "method": "POST",
                "path": path, "status_code": status_code, "duration_ms": 12.0}

    def test_normal_error_rate_no_anomaly(self):
        events = [self._event(200) for _ in range(19)] + [self._event(422)]
        result = analyze_errors(events, alert_rate=0.05)
        assert result["n_requests"] == 20
        assert result["error_rate"] == 0.05
        assert result["error_rate_anomaly"] is False

    def test_high_error_rate_flags_anomaly(self):
        events = ([self._event(200)] * 10
                  + [self._event(500)] * 5 + [self._event(422)] * 5)
        result = analyze_errors(events)
        assert result["error_rate"] == 0.5
        assert result["error_rate_anomaly"] is True
        assert result["errors_4xx"] == 5
        assert result["errors_5xx"] == 5

    def test_unhandled_exception_counts_as_5xx(self):
        result = analyze_errors([self._event(None), self._event(200)])
        assert result["errors_5xx"] == 1
        assert result["n_errors"] == 1

    def test_empty_request_log(self):
        result = analyze_errors([])
        assert result["n_requests"] == 0
        assert result["error_rate"] is None
        assert result["error_rate_anomaly"] is None

    def test_load_requests_log_jsonl(self, tmp_path):
        path = tmp_path / "requests.jsonl"
        lines = [
            json.dumps(self._event(200)),
            json.dumps(self._event(422, path="/predict")),
            "",  # trailing newline / blank line tolerated
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        events = load_requests_log(str(path))
        assert len(events) == 2
        assert events[0]["status_code"] == 200
        assert events[1]["status_code"] == 422


class TestDriftDetection:
    """Evidently-based drift detection on synthetic distributions."""

    def test_detects_shifted_column(self):
        pytest.importorskip("evidently")
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({
            "num_a": rng.normal(0, 1, 400),
            "num_b": rng.normal(0, 1, 400),
            "cat_a": rng.choice(["x", "y"], 400),
        })
        cur = pd.DataFrame({
            "num_a": rng.normal(5, 1, 200),  # shifted
            "num_b": rng.normal(0, 1, 200),   # stable
            "cat_a": rng.choice(["x", "y"], 200),
        })
        drift, _ = run_drift(ref, cur, FIELDS)
        by_name = {f["feature"]: f for f in drift["per_feature"]}
        assert by_name["num_a"]["drift_detected"] is True
        assert by_name["num_b"]["drift_detected"] is False
        assert drift["n_drifted_columns"] == 1

    def test_identical_distributions_no_drift(self):
        pytest.importorskip("evidently")
        rng = np.random.default_rng(1)
        ref = pd.DataFrame({
            "num_a": rng.normal(0, 1, 300),
            "num_b": rng.normal(0, 1, 300),
            "cat_a": rng.choice(["x", "y"], 300),
        })
        drift, _ = run_drift(ref, ref.copy(), FIELDS)
        assert drift["n_drifted_columns"] == 0
        assert drift["dataset_drift"] is False

    def test_dataset_drift_needs_half_the_columns(self):
        pytest.importorskip("evidently")
        rng = np.random.default_rng(2)
        ref = pd.DataFrame({
            "num_a": rng.normal(0, 1, 300),
            "num_b": rng.normal(0, 1, 300),
            "cat_a": rng.choice(["x", "y"], 300),
        })
        cur = pd.DataFrame({
            "num_a": rng.normal(6, 1, 300),  # shifted
            "num_b": rng.normal(0, 1, 300),
            "cat_a": rng.choice(["x", "y"], 300),
        })
        drift, _ = run_drift(ref, cur, FIELDS)
        # 1 of 3 columns drifted: per-column alert, no dataset-level drift
        assert drift["n_drifted_columns"] == 1
        assert drift["dataset_drift"] is False
