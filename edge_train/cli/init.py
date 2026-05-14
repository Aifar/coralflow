"""init command — download built-in datasets or prepare custom data."""

import sys
from pathlib import Path

import click

from edge_train.datasets import get_builtin, list_builtin, infer_modality


@click.group(name="init")
def init():
    """Download a built-in dataset or prepare custom data for training."""


@init.command(name="list")
def list_datasets():
    """List all available built-in datasets."""
    for name, info in list_builtin().items():
        modality = info.get("modality", "unknown")
        samples = info.get("samples", "?")
        desc = info.get("description", "")
        click.echo(f"  {name:<20} {modality:<8} {samples:>4} samples  {desc}")


@init.command()
@click.argument("dataset_name")
@click.option("--output", "-o", default="./data", help="Output directory")
def download(dataset_name: str, output: str):
    """Download a built-in dataset by name. Use `edge-train init list` to see options."""
    builtins = get_builtin()
    if dataset_name not in builtins:
        click.echo(f"Unknown dataset '{dataset_name}'. Available:", err=True)
        for name in builtins:
            click.echo(f"  {name}", err=True)
        sys.exit(1)

    info = builtins[dataset_name]
    out_dir = Path(output) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "data.csv"
    csv_path.write_text(info["csv_content"])
    click.echo(f"  Dataset '{dataset_name}' saved to {csv_path}")
    click.echo(f"  Modality: {info['modality']}")
    click.echo(f"  Samples: {info['samples']}")
    if info.get("classes"):
        click.echo(f"  Classes: {', '.join(info['classes'])}")


@init.command()
@click.argument("path", type=click.Path(exists=True))
def custom(path: str):
    """Prepare a custom dataset from a CSV file or image directory.

    The CLI will automatically detect the modality and validate the schema.
    """
    p = Path(path)
    if not p.is_file() and not p.is_dir():
        click.echo(f"Error: {path} is not a file or directory", err=True)
        sys.exit(1)

    if p.is_file():
        modality = infer_modality(p)
        click.echo(f"  Detected modality: {modality}")
        click.echo(f"  File: {p.name}")
        if modality == "text":
            _validate_text_csv(p)
        else:
            click.echo(f"  Modality '{modality}' support coming in a later release.")
    else:
        click.echo("  Directory datasets are not yet supported. Provide a CSV file.")
        sys.exit(1)


def _validate_text_csv(path: Path) -> None:
    """Basic validation of a text classification CSV."""
    import csv

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        click.echo("Error: CSV is empty", err=True)
        sys.exit(1)

    headers = reader.fieldnames or []
    text_cols = [c for c in headers if c.lower() in ("text", "message", "content", "sentence")]
    label_cols = [c for c in headers if c.lower() in ("label", "category", "class", "intent", "urgency")]

    if not text_cols and not label_cols:
        click.echo(
            "Error: Could not detect text and label columns. "
            "Expected columns named 'text'/'message' and 'label'/'category'.",
            err=True,
        )
        sys.exit(1)

    if not text_cols:
        click.echo("Warning: No obvious text column found. Using first column.", err=True)
        text_col = headers[0]
    else:
        text_col = text_cols[0]

    label_col = label_cols[0] if label_cols else headers[-1]
    label_counts: dict[str, int] = {}
    for row in rows:
        label = row.get(label_col, "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    click.echo(f"  Text column: '{text_col}'")
    click.echo(f"  Label column: '{label_col}'")
    click.echo(f"  Rows: {len(rows)}")
    click.echo(f"  Classes ({len(label_counts)}):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        click.echo(f"    {label}: {count}")

    # warn on severe imbalance
    ratios = [c / len(rows) for c in label_counts.values()]
    if ratios and max(ratios) > 0.8:
        click.echo("  Warning: Dataset is heavily imbalanced (>80% in one class).", err=True)
