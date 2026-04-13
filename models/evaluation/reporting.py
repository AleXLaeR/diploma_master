"""Evaluation artifact writing helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def prepare_output_dirs(base_output_dir: str | None = None) -> dict[str, Path]:
    """Create timestamped run directory and subfolders."""
    if base_output_dir:
        base = Path(base_output_dir)
    else:
        base = Path(__file__).resolve().parents[2] / "reports" / "evaluation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / timestamp
    figures_dir = run_dir / "figures"
    metrics_dir = run_dir / "metrics"
    for path in (figures_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base,
        "run_dir": run_dir,
        "figures_dir": figures_dir,
        "metrics_dir": metrics_dir,
    }


def write_metric_tables(metrics_dir: Path, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Write metric tables as CSV files. Returns relative filenames."""
    filenames: list[str] = []
    for name, frame in tables.items():
        path = metrics_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        filenames.append(path.name)
    return filenames


def write_summary_markdown(
    summary_path: Path,
    *,
    selected_folds: list[str],
    figure_status: dict[str, str],
    warnings: list[str],
) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Selected folds: `{', '.join(selected_folds)}`",
        "",
        "## Figure Status",
    ]
    for figure_id in sorted(figure_status.keys()):
        lines.append(f"- {figure_id}: {figure_status[figure_id]}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for item in warnings:
            lines.append(f"- {item}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    run_dir: Path,
    *,
    selected_folds: list[str],
    figure_status: dict[str, str],
    metric_files: list[str],
    warnings: list[str],
) -> Path:
    payload = {
        "selected_folds": selected_folds,
        "figures": figure_status,
        "metric_files": metric_files,
        "warnings": warnings,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def update_latest_pointer(base_dir: Path, manifest_path: Path) -> Path:
    latest_dir = base_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    target = latest_dir / "manifest.json"
    temp = latest_dir / "manifest.json.tmp"
    relative = manifest_path.relative_to(base_dir)
    temp.write_text(json.dumps({"manifest": str(relative)}, indent=2), encoding="utf-8")
    temp.replace(target)
    return target

