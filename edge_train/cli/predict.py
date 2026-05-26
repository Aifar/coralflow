"""predict command — run trained models on local text input."""

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
@click.option("--model", "-m", required=True, help="Path to SavedModel directory")
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
    model: str,
    text: str | None,
    csv_path: str | None,
    text_col: str | None,
    output: str | None,
    log_path: str | None,
):
    """Classify text using a locally trained model.

    Single:  edge-train predict --model ./model_output --text "hello world"
    Batch:   edge-train predict --model ./model_output --csv input.csv -o predictions.csv
    """
    from edge_train.inference import TextClassifier, log_prediction
    from edge_train.phoenix_util import ensure_phoenix_ready

    _, arize, train_cfg, _ = load_config()
    log_file = log_path or train_cfg.prediction_log_path

    phoenix_active = False
    if arize.is_valid():
        phoenix_active, phoenix_err = ensure_phoenix_ready(arize)
        if not phoenix_active:
            click.echo(phoenix_err, err=True)
            sys.exit(1)

    classifier = TextClassifier(model)

    if text is not None:
        label, conf = classifier.predict(text)
        probs = classifier.predict_proba(text)
        log_prediction(log_file, text, label, conf, probs, create_span=phoenix_active)
        click.echo(f"  Predicted: {label} ({conf:.4f})")
        if len(probs) > 1:
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1])[1:]:
                click.echo(f"    {cls}: {prob:.4f}")
        return

    if csv_path is not None:
        _predict_csv(classifier, csv_path, text_col, output, log_file, phoenix_active)
        return

    click.echo(
        "Error: provide --text for single prediction or --csv for batch.", err=True
    )
    sys.exit(1)


def _predict_csv(
    classifier,
    csv_path: str,
    text_col: str | None,
    output: str | None,
    log_path: str,
    phoenix_active: bool,
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
            log_path, texts[i], label, conf, probs, create_span=phoenix_active
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
