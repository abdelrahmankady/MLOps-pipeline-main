from __future__ import annotations

import json
import os
from typing import Dict, Any


def evaluate_model(
    model_uri: str,
    test_path: str | None = None,          # <- now optional
    metrics_path: str | None = None,
    data_paths: Dict[str, str] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simple evaluation wrapper for integration mode.

    We reuse pre-computed metrics saved by the ML engineering notebook
    instead of re-training / re-evaluating.

    Parameters
    ----------
    model_uri : str
        Path to the trained model JSON (kept for API compatibility).
    test_path : str | None
        Path to the test CSV (not used in this integration wrapper).
    metrics_path : str, optional
        Explicit path to metrics JSON. If None, defaults to
        <project_root>/model_artifacts/metrics.json.
    data_paths : dict, optional
        Extra argument passed by the Prefect flow (ignored here).
    **kwargs :
        Extra arguments for forward-compatibility (ignored).
    """

    # Project root = .../MLOPS-pipeline
    project_root = os.path.dirname(os.path.dirname(__file__))

    if metrics_path is None:
        metrics_path = os.path.join(project_root, "model_artifacts", "metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at:\n  {metrics_path}\n"
            "Make sure metrics.json is under MLOPS-pipeline/model_artifacts/."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics: Dict[str, Any] = json.load(f)

    print("Loaded precomputed metrics from:", metrics_path)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics
