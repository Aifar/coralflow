"""Local hardware resource assessment for training eligibility.

Uses psutil for cross-platform CPU, RAM, and disk detection.
Estimates dataset memory footprint and returns a structured verdict.
"""

from __future__ import annotations

import os
from pathlib import Path


def assess_resources(dataset_path: str | None = None) -> str:
    """Check local hardware and return a structured eligibility report.

    Parameters:
        dataset_path: Optional path to a CSV dataset for memory estimation.

    Returns:
        Formatted text report with verdict and recommendation.
    """
    import psutil

    lines = ["Local Resource Assessment:"]

    # CPU
    cpu_count = psutil.cpu_count(logical=True) or 1
    cpu_ok = cpu_count >= 2
    lines.append(
        f"  CPU: {cpu_count} cores — {'OK' if cpu_ok else 'marginal (2+ recommended)'}"
    )

    # RAM
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    ram_ok = avail_gb >= 1.0
    lines.append(
        f"  RAM: {avail_gb:.1f} GB available — {'OK' if ram_ok else 'INSUFFICIENT (need ≥1 GB)'}"
    )

    # Disk
    cwd = os.getcwd()
    try:
        disk = psutil.disk_usage(cwd)
        free_gb = disk.free / (1024**3)
        disk_ok = free_gb >= 0.2
        lines.append(
            f"  Disk: {free_gb:.1f} GB free on {cwd} — {'OK' if disk_ok else 'tight (need ≥200 MB)'}"
        )
    except Exception:
        lines.append("  Disk: (could not check)")

    # TensorFlow
    try:
        import tensorflow as tf

        lines.append(f"  TensorFlow: {tf.__version__} — OK")
    except ImportError:
        lines.append("  TensorFlow: not installed — REQUIRED for local training")

    # Dataset memory estimate
    if (
        dataset_path
        and not dataset_path.startswith("builtin:")
        and Path(dataset_path).exists()
    ):
        try:
            import csv

            with open(dataset_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # Rough: ~1 KB per text row for tokenized representation
            est_mb = len(rows) * 1 / 1024
            fits = est_mb < (
                avail_gb * 512
            )  # dataset should fit in 50% of available RAM
            lines.append(
                f"  Dataset fit: {len(rows)} rows → ~{est_mb:.1f} MB in memory — "
                f"{'OK' if fits else 'borderline — may OOM at current RAM'}"
            )
        except Exception:
            pass

    lines.append("")

    # Verdict
    if ram_ok and cpu_ok:
        lines.append("Verdict: Local training viable")
        lines.append("Options:")
        lines.append("  [1] Local training — TF Keras, free, fast for small datasets")
        lines.append("  [2] Cloud training — Vertex AI AutoML, ~$3-8, 30-60 min")
    else:
        lines.append("Verdict: Local resources insufficient for reliable training")
        lines.append("Recommendation: Google Cloud Vertex AI")
        reasons = []
        if not ram_ok:
            reasons.append(f"available RAM ({avail_gb:.1f} GB) below 1 GB minimum")
        if not cpu_ok:
            reasons.append(f"only {cpu_count} CPU core(s), 2+ recommended")
        lines.append(
            f"  Rationale: {'; '.join(reasons)}. Training would likely OOM-kill the process."
        )
        lines.append("  Estimated Cloud cost: ~$3-8 for typical dataset sizes.")

    return "\n".join(lines)
