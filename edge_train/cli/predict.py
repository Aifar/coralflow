"""predict command — local SavedModel or Vertex endpoint inference."""

import csv
import sys

import click

from edge_train.config import load_config

TEXT_COLUMN_NAMES = {
    "text",
    "message",
    "content",
    "sentence",
    "review",
    "comment",
    "description",
}


def _detect_text_column(csv_path: str) -> str | None:
    with open(csv_path, encoding="utf-8") as f:
        headers = csv.DictReader(f).fieldnames or []
    for h in headers:
        if h.lower().strip() in TEXT_COLUMN_NAMES:
            return h
    return headers[0] if headers else None


@click.command()
@click.option(
    "--model",
    "-m",
    default=None,
    help="Path to local SavedModel directory",
)
@click.option(
    "--endpoint",
    "-e",
    default=None,
    help="Vertex AI endpoint resource name (Gemini fine-tuned text models)",
)
@click.option("--text", "-t", default=None, help="Single text input to classify")
@click.option(
    "--csv", "-c", "csv_path", default=None, help="CSV file for batch prediction"
)
@click.option(
    "--text-col", default=None, help="Column name for text in CSV (auto-detected)"
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output CSV path (batch mode; stdout if omitted)",
)
@click.option("--log", "-l", "log_path", default=None, help="Prediction log file path")
def predict(
    model: str | None,
    endpoint: str | None,
    text: str | None,
    csv_path: str | None,
    text_col: str | None,
    output: str | None,
    log_path: str | None,
):
    """Classify text using a local model or a Vertex AI endpoint.

    Local:    coralflow predict --model ./model_output --text "hello world"
    Vertex:   coralflow predict --endpoint projects/.../endpoints/ID --text "hello"

    Both paths log to prediction_log.jsonl and send OTEL spans to Arize Phoenix when configured.
    """
    if bool(model) == bool(endpoint):
        click.echo(
            "Error: provide exactly one of --model (local) or --endpoint (Vertex).",
            err=True,
        )
        sys.exit(1)

    from edge_train.inference.phoenix import (
        echo_phoenix_exit_error,
        prepare_phoenix_for_inference,
    )

    _, arize, train_cfg, _ = load_config()
    log_file = log_path or train_cfg.prediction_log_path

    phoenix_active = False
    if arize.is_valid():
        phoenix_active, phoenix_err = prepare_phoenix_for_inference(required=True)
        if not phoenix_active:
            echo_phoenix_exit_error(phoenix_err)

    if endpoint:
        classifier = _load_vertex_predictor(endpoint)
        source = "vertex"
    else:
        from edge_train.inference import TextClassifier

        classifier = TextClassifier(model)
        source = "local"

    if text is not None:
        _predict_single(classifier, text, log_file, phoenix_active, source=source)
        return

    if csv_path is not None:
        _predict_csv(
            classifier, csv_path, text_col, output, log_file, phoenix_active, source
        )
        return

    click.echo(
        "Error: provide --text for single prediction or --csv for batch.", err=True
    )
    sys.exit(1)


def _load_vertex_predictor(endpoint: str):
    from edge_train.cloud.serving import VertexTextPredictor
    from edge_train.config import GCPConfig

    gcp = GCPConfig()
    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT for Vertex endpoint predict.",
            err=True,
        )
        sys.exit(1)
    return VertexTextPredictor(endpoint, project=gcp.project_id, location=gcp.location)


def _predict_single(
    classifier, text: str, log_path: str, phoenix_active: bool, *, source: str
):
    from edge_train.inference import log_prediction

    label, conf = classifier.predict(text)
    probs = classifier.predict_proba(text)
    log_prediction(
        log_path,
        text,
        label,
        conf,
        probs,
        create_span=phoenix_active,
        source=source,
    )
    click.echo(f"  Predicted: {label} ({conf:.4f})")
    if len(probs) > 1:
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1])[1:]:
            click.echo(f"    {cls}: {prob:.4f}")
    if phoenix_active:
        click.echo("  OTEL span sent to Arize Phoenix.")


def _predict_csv(
    classifier,
    csv_path: str,
    text_col: str | None,
    output: str | None,
    log_path: str,
    phoenix_active: bool,
    source: str,
):
    from edge_train.inference import log_prediction

    col = text_col or _detect_text_column(csv_path)
    if col is None:
        click.echo(
            "Error: could not auto-detect text column. Use --text-col.", err=True
        )
        sys.exit(1)

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    texts = [r[col] for r in rows]
    results = classifier.predict_batch(texts)

    out_rows = []
    for i, row in enumerate(rows):
        label, conf = results[i]
        probs = classifier.predict_proba(texts[i])
        out_rows.append({**row, "predicted_label": label, "confidence": round(conf, 4)})
        log_prediction(
            log_path,
            texts[i],
            label,
            conf,
            probs,
            create_span=phoenix_active,
            source=source,
        )

    if output:
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        click.echo(f"  Predictions saved to: {output}")
    else:
        for r in out_rows:
            click.echo(f"  {r[col]}: {r['predicted_label']} ({r['confidence']:.4f})")

    click.echo(f"  Logged {len(results)} predictions to: {log_path}")
    if phoenix_active:
        click.echo(f"  Sent {len(results)} OTEL spans to Arize Phoenix.")
