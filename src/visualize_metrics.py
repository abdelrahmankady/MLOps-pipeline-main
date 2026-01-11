from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics() -> dict:
    """Load metrics from project_root/model_artifacts/metrics.json."""
    # project_root = root/
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "model_artifacts" / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json not found at:\n  {metrics_path}\n"
            "Make sure you ran the pipeline and that metrics.json is under model_artifacts/."
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def plot_bar(metrics: dict, output_dir: Path) -> Path:
    """Create a bar chart for accuracy / precision / recall / f1."""
    # keys we expect
    possible_keys = ["accuracy", "precision", "recall", "f1", "f1_macro", "macro_f1"]

    # filter only existing metrics
    labels = []
    values = []
    for key in possible_keys:
        if key in metrics:
            labels.append(key)
            values.append(float(metrics[key]))

    if not labels:
        raise ValueError("No known metric keys found in metrics.json.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics_bar.png"

    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Evaluation Metrics")

    # add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def main() -> None:
    # project_root = root/
    project_root = Path(__file__).resolve().parents[1]

    # charts will be saved in: <root>/model_artifacts/plots/
    reports_dir = project_root / "model_artifacts" / "plots"

    metrics = load_metrics()
    print("Loaded metrics from metrics.json:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    img_path = plot_bar(metrics, reports_dir)
    print(f"\n✅ Metrics bar chart saved to:\n  {img_path}")


if __name__ == "__main__":
    main()
