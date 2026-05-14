"""cost command — estimate Vertex AI training costs before running."""

import click
from edge_train.datasets import list_builtin


@click.command()
@click.argument("dataset_name", required=False)
@click.option("--hours", type=float, default=None,
              help="Estimated training hours (overrides auto-estimate)")
def cost(dataset_name: str | None, hours: float | None):
    """Estimate Vertex AI training costs for a dataset.

    Uses AutoML pricing (~$20-50/hr depending on modality) and
    estimated dataset size to project training cost.
    """
    # Vertex AI AutoML pricing (approximate, per hour)
    PRICING = {
        "text": 40.0,
        "image": 25.0,
        "table": 50.0,
    }

    if dataset_name:
        builtins = list_builtin()
        if dataset_name not in builtins:
            click.echo(f"Unknown dataset '{dataset_name}'. Use 'edge-train init list' to see options.")
            return

        info = builtins[dataset_name]
        modality = info.get("modality", "text")
        samples = info.get("samples", 0)
        rate = PRICING.get(modality, 40.0)
        est_hours = hours or _estimate_hours(modality, samples)
    else:
        click.echo("  edge-train cost estimator")
        click.echo("")
        click.echo("  Hourly rates (Vertex AI AutoML):")
        for mod, rate in PRICING.items():
            click.echo(f"    {mod:<8} ${rate:.0f}/hr")
        click.echo("")
        click.echo("  Example — built-in text datasets (~5 min training):")
        click.echo(f"    ~$3.30 per training run")
        click.echo("")
        click.echo("  Usage: edge-train cost <dataset-name>")
        return

    total = rate * est_hours
    click.echo(f"  Dataset: {dataset_name}")
    click.echo(f"  Modality: {modality}")
    click.echo(f"  Samples: {samples}")
    click.echo(f"  Estimated training time: {est_hours:.2f} hrs")
    click.echo(f"  Estimated cost: ${total:.2f}")
    click.echo("")
    click.echo("  Note: New GCP accounts get $300 free credits.")


def _estimate_hours(modality: str, samples: int) -> float:
    """Rough training time estimate based on modality and sample count."""
    base = {"text": 0.08, "image": 0.12, "table": 0.05}.get(modality, 0.1)
    return base * (samples / 500)
