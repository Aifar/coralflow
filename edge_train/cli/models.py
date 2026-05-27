"""models command — list Vertex AI publisher and custom models."""

import sys

import click

from edge_train.config import ensure_gcp_credentials, load_config


@click.group()
def models():
    """Inspect Vertex AI models (publisher foundation vs project custom)."""


@models.command("list")
@click.option(
    "--filter",
    "model_filter",
    default="gemini",
    help="Keyword filter for Model Garden (default: gemini)",
)
@click.option(
    "--all-models",
    "show_all",
    is_flag=True,
    help="Include non-text models (embedding, image, TTS, etc.)",
)
@click.option(
    "--custom",
    "show_custom",
    is_flag=True,
    help="Also list custom models in your project's Model Registry",
)
def list_models(model_filter: str, show_all: bool, show_custom: bool):
    """List Google publisher (foundation) models available for cloud training."""
    gcp, _, _, _ = load_config()

    ok, err = ensure_gcp_credentials()
    if not ok:
        click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    if not gcp.project_id:
        click.echo("Error: GCP_PROJECT is not set.", err=True)
        sys.exit(1)

    from edge_train.cloud.publisher_models import (
        list_custom_models,
        list_publisher_models,
    )

    click.echo(
        "Publisher models (Model Garden — Google foundation models, not your project):"
    )
    click.echo(f"  Project: {gcp.project_id}")
    click.echo(f"  Location: {gcp.location}")
    if model_filter:
        click.echo(f"  Filter: {model_filter}")

    try:
        publisher = list_publisher_models(
            gcp.project_id,
            gcp.location,
            model_filter=model_filter or None,
            finetune_capable_only=not show_all,
        )
    except Exception as exc:
        click.echo(f"Error listing publisher models: {exc}", err=True)
        sys.exit(1)

    if not publisher:
        click.echo("  (no models matched)")
    else:
        for model in publisher:
            click.echo(f"  {model.model_id}")
            click.echo(f"    qualified: {model.qualified_name}")
            click.echo(f"    resource:  {model.resource_name}")
            click.echo(f"    SFT source_model: {model.tuning_source_model}")

    if show_custom:
        click.echo("")
        click.echo("Custom models (your project's Model Registry):")
        try:
            custom = list_custom_models(gcp.project_id, gcp.location)
        except Exception as exc:
            click.echo(f"  Error: {exc}", err=True)
            sys.exit(1)

        if not custom:
            click.echo(
                "  (empty — expected for new projects; appears after you train/save a model)"
            )
        else:
            for name in custom:
                click.echo(f"  {name}")
