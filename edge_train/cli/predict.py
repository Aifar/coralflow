"""predict command — local SavedModel or Vertex endpoint inference."""

import csv
import json
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

URI_COLUMN_NAMES = {
    "gcs_uri",
    "gcsuri",
    "gcs",
    "uri",
    "video_uri",
    "video",
}

IMAGE_COLUMN_NAMES = {
    "image",
    "image_path",
    "imagepath",
    "path",
    "file",
    "filename",
}


def _detect_text_column(csv_path: str) -> str | None:
    with open(csv_path, encoding="utf-8") as f:
        headers = csv.DictReader(f).fieldnames or []
    for h in headers:
        if h.lower().strip() in TEXT_COLUMN_NAMES:
            return h
    return headers[0] if headers else None


def _detect_uri_column(headers: list[str]) -> str | None:
    for h in headers:
        if h.lower().strip() in URI_COLUMN_NAMES:
            return h
    for h in headers:
        if h.lower().strip().startswith("gs://") or "gcs" in h.lower():
            return h
    return None


def _detect_image_column(headers: list[str]) -> str | None:
    for h in headers:
        if h.lower().strip() in IMAGE_COLUMN_NAMES:
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
    help="Vertex AI endpoint resource name",
)
@click.option(
    "--modality",
    type=click.Choice(["text", "table", "image", "video"]),
    default=None,
    help="Vertex endpoint modality (auto-detected from deployment registry when possible)",
)
@click.option("--text", "-t", default=None, help="Single text input (text modality)")
@click.option(
    "--features",
    default=None,
    help='Tabular feature JSON for single predict, e.g. \'{"age": 25, "plan": "monthly"}\'',
)
@click.option("--image", default=None, help="Local image path (image modality)")
@click.option(
    "--gcs-uri",
    default=None,
    help="GCS URI for image or video input (gs://bucket/object)",
)
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
    modality: str | None,
    text: str | None,
    features: str | None,
    image: str | None,
    gcs_uri: str | None,
    csv_path: str | None,
    text_col: str | None,
    output: str | None,
    log_path: str | None,
):
    """Classify using a local model or a Vertex AI endpoint.

    Local text:  coralflow predict --model ./model_output --text "hello"
    Vertex text: coralflow predict --endpoint projects/.../endpoints/ID --text "hello"
    Vertex table: coralflow predict --endpoint ... --modality table --csv rows.csv
    Vertex image: coralflow predict --endpoint ... --modality image --image cat.jpg
    Vertex video: coralflow predict --endpoint ... --modality video --gcs-uri gs://.../clip.mp4

    All paths log to prediction_log.jsonl and send OTEL spans to Arize Phoenix when configured.
    """
    if bool(model) == bool(endpoint):
        click.echo(
            "Error: provide exactly one of --model (local) or --endpoint (Vertex).",
            err=True,
        )
        sys.exit(1)

    from edge_train.inference.phoenix import (
        apply_phoenix_prepare,
        prepare_phoenix_for_inference,
    )

    _, arize, train_cfg, _ = load_config()
    log_file = log_path or train_cfg.prediction_log_path

    phoenix_active = apply_phoenix_prepare(
        prepare_phoenix_for_inference(required=arize.is_valid(), interactive=True)
    )

    if endpoint:
        resolved_modality = _resolve_endpoint_modality(endpoint, modality)
        classifier = _load_vertex_predictor(endpoint, resolved_modality)
        source = "vertex"
        _run_vertex_predict(
            classifier,
            resolved_modality,
            text=text,
            features=features,
            image=image,
            gcs_uri=gcs_uri,
            csv_path=csv_path,
            text_col=text_col,
            output=output,
            log_file=log_file,
            phoenix_active=phoenix_active,
            source=source,
        )
        return

    from edge_train.inference import TextClassifier

    classifier = TextClassifier(model)
    source = "local"

    if text is not None:
        _predict_single(
            classifier, text, log_file, phoenix_active, source=source, input_label=text
        )
        return

    if csv_path is not None:
        _predict_text_csv(
            classifier, csv_path, text_col, output, log_file, phoenix_active, source
        )
        return

    click.echo(
        "Error: provide --text for single prediction or --csv for batch.", err=True
    )
    sys.exit(1)


def _resolve_endpoint_modality(endpoint: str, modality: str | None) -> str:
    if modality:
        return modality
    from edge_train.deployments import DeploymentRegistry

    record = DeploymentRegistry.load().find_by_endpoint(endpoint)
    if record and record.modality:
        return record.modality
    return "text"


def _load_vertex_predictor(endpoint: str, modality: str):
    from edge_train.cloud.serving import resolve_vertex_predictor
    from edge_train.config import GCPConfig

    gcp = GCPConfig()
    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT for Vertex endpoint predict.",
            err=True,
        )
        sys.exit(1)
    return resolve_vertex_predictor(
        endpoint,
        project=gcp.project_id,
        location=gcp.location,
        modality=modality,
    )


def _run_vertex_predict(
    classifier,
    modality: str,
    *,
    text: str | None,
    features: str | None,
    image: str | None,
    gcs_uri: str | None,
    csv_path: str | None,
    text_col: str | None,
    output: str | None,
    log_file: str,
    phoenix_active: bool,
    source: str,
) -> None:
    mod = modality.lower()
    if mod == "text":
        if text is not None:
            _predict_single(
                classifier,
                text,
                log_file,
                phoenix_active,
                source=source,
                input_label=text,
            )
            return
        if csv_path is not None:
            _predict_text_csv(
                classifier, csv_path, text_col, output, log_file, phoenix_active, source
            )
            return
        click.echo(
            "Error: text endpoint requires --text or --csv with a text column.",
            err=True,
        )
        sys.exit(1)

    if mod == "table":
        if features is not None:
            try:
                row = json.loads(features)
            except json.JSONDecodeError as exc:
                click.echo(f"Error: invalid --features JSON: {exc}", err=True)
                sys.exit(1)
            if not isinstance(row, dict):
                click.echo("Error: --features must be a JSON object.", err=True)
                sys.exit(1)
            _predict_single(
                classifier,
                row,
                log_file,
                phoenix_active,
                source=source,
                input_label=classifier.format_input(row),
            )
            return
        if csv_path is not None:
            _predict_tabular_csv(
                classifier, csv_path, output, log_file, phoenix_active, source
            )
            return
        click.echo(
            "Error: table endpoint requires --features JSON or --csv with feature columns.",
            err=True,
        )
        sys.exit(1)

    if mod == "image":
        payload = image or gcs_uri
        if payload and csv_path is None:
            _predict_single(
                classifier,
                payload,
                log_file,
                phoenix_active,
                source=source,
                input_label=classifier.format_input(payload),
            )
            return
        if csv_path is not None:
            _predict_uri_csv(
                classifier,
                csv_path,
                output,
                log_file,
                phoenix_active,
                source,
                column_kind="image",
            )
            return
        click.echo(
            "Error: image endpoint requires --image, --gcs-uri, or --csv with image paths.",
            err=True,
        )
        sys.exit(1)

    if mod == "video":
        payload = gcs_uri or image
        if payload and csv_path is None:
            _predict_single(
                classifier,
                payload,
                log_file,
                phoenix_active,
                source=source,
                input_label=classifier.format_input(payload),
            )
            return
        if csv_path is not None:
            _predict_uri_csv(
                classifier,
                csv_path,
                output,
                log_file,
                phoenix_active,
                source,
                column_kind="video",
            )
            return
        click.echo(
            "Error: video endpoint requires --gcs-uri or --csv with GCS URIs.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Error: unsupported modality {modality}.", err=True)
    sys.exit(1)


def _predict_single(
    classifier,
    payload,
    log_path: str,
    phoenix_active: bool,
    *,
    source: str,
    input_label: str,
):
    from edge_train.inference import log_prediction

    label, conf = classifier.predict(payload)
    probs = classifier.predict_proba(payload)
    log_prediction(
        log_path,
        input_label,
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


def _predict_text_csv(
    classifier,
    csv_path: str,
    text_col: str | None,
    output: str | None,
    log_path: str,
    phoenix_active: bool,
    source: str,
):
    col = text_col or _detect_text_column(csv_path)
    if col is None:
        click.echo(
            "Error: could not auto-detect text column. Use --text-col.", err=True
        )
        sys.exit(1)
    _predict_csv_column(
        classifier, csv_path, col, output, log_path, phoenix_active, source
    )


def _predict_tabular_csv(
    classifier,
    csv_path: str,
    output: str | None,
    log_path: str,
    phoenix_active: bool,
    source: str,
):
    rows = _read_csv_rows(csv_path)
    payloads = [dict(row) for row in rows]
    results = classifier.predict_batch(payloads)

    out_rows = []
    for i, row in enumerate(rows):
        label, conf = results[i]
        probs = classifier.predict_proba(payloads[i])
        display = classifier.format_input(payloads[i])
        out_rows.append({**row, "predicted_label": label, "confidence": round(conf, 4)})
        _log_row(log_path, display, label, conf, probs, phoenix_active, source)

    _emit_batch_output(out_rows, output, log_path, len(results), phoenix_active)


def _predict_uri_csv(
    classifier,
    csv_path: str,
    output: str | None,
    log_path: str,
    phoenix_active: bool,
    source: str,
    *,
    column_kind: str,
):
    rows = _read_csv_rows(csv_path)
    headers = list(rows[0].keys()) if rows else []
    if column_kind == "video":
        col = _detect_uri_column(headers)
    else:
        col = _detect_image_column(headers)
    if col is None:
        click.echo(
            f"Error: could not auto-detect {column_kind} column in CSV.", err=True
        )
        sys.exit(1)

    payloads = [row[col] for row in rows]
    results = classifier.predict_batch(payloads)

    out_rows = []
    for i, row in enumerate(rows):
        label, conf = results[i]
        probs = classifier.predict_proba(payloads[i])
        display = classifier.format_input(payloads[i])
        out_rows.append({**row, "predicted_label": label, "confidence": round(conf, 4)})
        _log_row(log_path, display, label, conf, probs, phoenix_active, source)

    _emit_batch_output(out_rows, output, log_path, len(results), phoenix_active)


def _predict_csv_column(
    classifier,
    csv_path: str,
    col: str,
    output: str | None,
    log_path: str,
    phoenix_active: bool,
    source: str,
):
    rows = _read_csv_rows(csv_path)
    payloads = [row[col] for row in rows]
    results = classifier.predict_batch(payloads)

    out_rows = []
    for i, row in enumerate(rows):
        label, conf = results[i]
        probs = classifier.predict_proba(payloads[i])
        display = (
            classifier.format_input(payloads[i])
            if hasattr(classifier, "format_input")
            else str(payloads[i])
        )
        out_rows.append({**row, "predicted_label": label, "confidence": round(conf, 4)})
        _log_row(log_path, display, label, conf, probs, phoenix_active, source)

    _emit_batch_output(out_rows, output, log_path, len(results), phoenix_active)


def _read_csv_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _log_row(
    log_path: str,
    display: str,
    label: str,
    conf: float,
    probs: dict,
    phoenix_active: bool,
    source: str,
) -> None:
    from edge_train.inference import log_prediction

    log_prediction(
        log_path,
        display,
        label,
        conf,
        probs,
        create_span=phoenix_active,
        source=source,
    )


def _emit_batch_output(
    out_rows: list[dict],
    output: str | None,
    log_path: str,
    count: int,
    phoenix_active: bool,
) -> None:
    if not out_rows:
        click.echo("  No rows to predict.", err=True)
        sys.exit(1)
    if output:
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        click.echo(f"  Predictions saved to: {output}")
    else:
        for r in out_rows:
            click.echo(
                f"  {r.get('predicted_label', '?')}: {r.get('confidence', 0):.4f}"
            )
    click.echo(f"  Logged {count} predictions to: {log_path}")
    if phoenix_active:
        click.echo(f"  Sent {count} OTEL spans to Arize Phoenix.")
