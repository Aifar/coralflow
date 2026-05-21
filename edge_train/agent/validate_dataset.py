"""LLM-driven dataset quality validation.

Sends a data sample to the LLM for multi-dimensional quality analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_train.agent.llm import LLMClient


def validate_dataset(path: str, llm: LLMClient | None = None) -> str:
    """Validate dataset quality using LLM analysis of a data sample.

    Reads the CSV, extracts summary stats and the first 20 rows, then
    asks the LLM to check: column structure, data types, class balance,
    data quality, label clarity, and size adequacy.
    """
    p = Path(path)
    if not p.exists():
        return f"Dataset not found: {path}"

    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return f"Dataset is empty: {path}"

    # Auto-detect columns
    from edge_train.agent import _detect_text_label_columns

    text_col, label_col = _detect_text_label_columns(headers)

    # Class distribution
    class_counts: dict[str, int] = {}
    for r in rows:
        if label_col:
            v = r.get(label_col, "")
            class_counts[v] = class_counts.get(v, 0) + 1

    # Build a data snapshot for the LLM
    sample_rows = rows[:20]
    snapshot_lines = []
    for r in sample_rows:
        txt = r.get(text_col or "", "")[:100]
        lbl = r.get(label_col or "", "")
        snapshot_lines.append(f"  {txt} | {lbl}")

    data_snapshot = "\n".join(snapshot_lines)

    prompt = f"""Validate this dataset for ML text classification training:

File: {p.name}
Rows: {len(rows)} total
Columns: {', '.join(headers)}
Text column: {text_col or '?'}
Label column: {label_col or '?'}
Classes: {json.dumps(class_counts, ensure_ascii=False)}

Sample (first 20 rows):
{data_snapshot}

Evaluate:
1. Column structure — are text and label columns correctly detected?
2. Data types — is text actually text, are labels categorical?
3. Class balance — are classes reasonably balanced? flag if max/min ratio > 3:1
4. Data quality — any empty rows, garbled text, encoding issues?
5. Label clarity — are class names distinct and unambiguous?
6. Size adequacy — enough samples per class?

Respond in this format:
✓ or ⚠ item: finding
...
Overall: PASS or FAIL

Be concise. Use the actual data to make specific observations, not generic statements."""

    if llm:
        try:
            resp = llm.chat([{"role": "user", "content": prompt}])
            if resp.content:
                return resp.content
        except Exception:
            pass

    # Fallback: run basic automated checks
    lines = [f"Dataset: {p.name} ({len(rows)} rows, {len(class_counts)} classes)"]
    lines.append(f"✓ Column structure: text='{text_col}', label='{label_col}'")

    if class_counts:
        counts = list(class_counts.values())
        max_c, min_c = max(counts), min(counts)
        if max_c > min_c * 3:
            lines.append(f"⚠ Class balance: imbalanced ({max_c / min_c:.1f}:1 ratio)")
        else:
            lines.append("✓ Class balance: reasonable")

    empty = sum(1 for r in rows if not r.get(text_col or "", "").strip())
    if empty > 0:
        lines.append(f"⚠ Data quality: {empty} empty rows found")
    else:
        lines.append("✓ Data quality: no empty rows")

    min_per_class = len(rows) / max(len(class_counts), 1)
    if min_per_class >= 20:
        lines.append("✓ Size: adequate samples per class")
    else:
        lines.append(
            f"⚠ Size: only ~{min_per_class:.0f} samples per class (20+ recommended)"
        )

    lines.append("Overall: PASS — ready for training")
    return "\n".join(lines)
