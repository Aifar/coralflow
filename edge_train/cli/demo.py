"""demo command — runnable end-to-end flows for presentations and CI."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from edge_train.demo.challenges import (
    URGENT_DRIFT_CASES,
    URGENT_HARD_TRAIN_LABELS,
    write_urgent_challenge_csv,
)
from edge_train.retrain import (
    compute_accuracy,
    evaluate_cases,
    read_prediction_log,
    retrain_from_labeled,
    write_prediction_log,
)


def _resolve_dataset(dataset: str, work_dir: Path) -> Path:
    if dataset.startswith("builtin:"):
        name = dataset.split(":", 1)[1]
        from edge_train.datasets import get_builtin

        builtin = get_builtin().get(name)
        if not builtin:
            raise click.ClickException(f"Unknown builtin dataset: {name}")
        path = work_dir / f"{name}.csv"
        path.write_text(builtin["csv_content"], encoding="utf-8")
        return path
    path = Path(dataset)
    if not path.exists():
        raise click.ClickException(f"Dataset not found: {dataset}")
    return path


def _print_case_table(details: list[dict], limit: int = 8) -> None:
    shown = details[:limit]
    for i, row in enumerate(shown, 1):
        mark = "✓" if row["correct"] else "✗"
        text = row["text"]
        if len(text) > 36:
            text = text[:33] + "..."
        click.echo(
            f"    {mark} [{i}] pred={row['predicted_label']:<4} "
            f"truth={row['ground_truth']:<4} conf={row['confidence']:.2f}  {text}"
        )
    if len(details) > limit:
        click.echo(f"    ... and {len(details) - limit} more")


@click.group()
def demo():
    """Demonstration workflows (retrain loop, drift simulation)."""


@demo.command("retrain-loop")
@click.option(
    "--dataset",
    "-d",
    default="builtin:urgent",
    show_default=True,
    help="Training CSV path or builtin:urgent / builtin:expense",
)
@click.option(
    "--work-dir",
    "-w",
    default="./demo_retrain",
    show_default=True,
    help="Directory for models, logs, and intermediate CSVs",
)
@click.option(
    "--epochs",
    default=3,
    show_default=True,
    type=int,
    help="Epochs for the initial (weak) baseline model",
)
@click.option(
    "--retrain-epochs",
    default=10,
    show_default=True,
    type=int,
    help="Epochs when retraining after drift simulation",
)
@click.option(
    "--hard/--no-hard",
    default=True,
    show_default=True,
    help="Train baseline without 紧急/重要 labels so challenge cases mispredict",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Retrain accuracy threshold (default from config, usually 0.85)",
)
@click.option(
    "--log",
    "log_path",
    default=None,
    help="Prediction log path (default: <work-dir>/prediction_log.jsonl)",
)
def retrain_loop(
    dataset: str,
    work_dir: str,
    epochs: int,
    retrain_epochs: int,
    hard: bool,
    threshold: float | None,
    log_path: str | None,
):
    """Simulate drift: mispredict challenge data → retrain → show higher accuracy.

    Steps:
      1. Train a baseline model (by default on 一般/可忽略 only).
      2. Run challenge phrases (紧急/重要 paraphrases) through predict → log.
      3. If log accuracy < threshold, merge labels into training data and retrain.
      4. Re-evaluate the same challenge set on the new model.
    """
    from edge_train.config import load_config
    from edge_train.inference import TextClassifier
    from edge_train.trainer import train_text_classifier
    from edge_train.retrain import filter_csv_by_labels

    _, _, train_cfg, _ = load_config()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    full_csv = _resolve_dataset(dataset, work)
    challenge_csv = write_urgent_challenge_csv(work / "drift_challenge.csv")
    log_file = Path(log_path or work / "prediction_log.jsonl")
    acc_threshold = (
        threshold if threshold is not None else train_cfg.retrain_accuracy_threshold
    )

    click.echo("\n  CoralFlow retrain-loop demo")
    click.echo(f"  Work dir:     {work.resolve()}")
    click.echo(f"  Dataset:      {full_csv}")
    click.echo(f"  Challenge:    {len(URGENT_DRIFT_CASES)} samples → {challenge_csv}")
    click.echo(f"  Hard baseline: {hard}")
    click.echo("")

    # ── 1. Baseline train ─────────────────────────────────────────────
    if hard:
        train_csv = work / "train_partial.csv"
        n = filter_csv_by_labels(
            str(full_csv), str(train_csv), URGENT_HARD_TRAIN_LABELS
        )
        click.echo(
            f"  [1/4] Training baseline on labels {URGENT_HARD_TRAIN_LABELS} only "
            f"({n} rows, {epochs} epochs)..."
        )
    else:
        train_csv = full_csv
        click.echo(f"  [1/4] Training baseline on full dataset ({epochs} epochs)...")

    model_v0 = work / "model_v0"
    train_text_classifier(
        dataset_path=str(train_csv),
        output_dir=str(model_v0),
        epochs=epochs,
    )
    click.echo(f"        Baseline model: {model_v0}")

    # ── 2. Simulate production traffic (challenge set) ────────────────
    click.echo(
        "\n  [2/4] Running challenge set through baseline (logging ground truth)..."
    )
    clf0 = TextClassifier(str(model_v0))
    write_prediction_log(log_file, URGENT_DRIFT_CASES, clf0, clear=True)

    acc0, details0 = evaluate_cases(clf0, URGENT_DRIFT_CASES)
    log_entries = read_prediction_log(log_file)
    log_acc, log_correct, log_total = compute_accuracy(log_entries)

    click.echo(
        f"        Challenge accuracy (baseline): {acc0:.1%} ({log_correct}/{log_total})"
    )
    _print_case_table(details0)

    # ── 3. Retrain if below threshold ─────────────────────────────────
    click.echo(
        f"\n  [3/4] Retrain check (threshold {acc_threshold:.0%}, "
        f"min {train_cfg.retrain_min_samples} labeled samples)..."
    )

    if log_total < train_cfg.retrain_min_samples:
        raise click.ClickException(
            f"Need at least {train_cfg.retrain_min_samples} challenge samples in log."
        )

    if log_acc >= acc_threshold:
        click.echo(
            f"        Accuracy {log_acc:.1%} is already ≥ threshold — "
            "retrain would not trigger."
        )
        click.echo(
            "        Tip: use --hard (default) or lower --epochs for a weaker baseline."
        )
        sys.exit(1)

    click.echo(f"        Accuracy {log_acc:.1%} < {acc_threshold:.0%} — retraining...")
    labeled = [e for e in log_entries if e.get("ground_truth") is not None]
    model_v1 = retrain_from_labeled(
        labeled,
        str(full_csv),
        str(work / "model_retrained"),
        retrain_epochs,
        oversample=3,
    )
    click.echo(f"        Retrained model: {model_v1}")

    # ── 4. Re-evaluate ────────────────────────────────────────────────
    click.echo("\n  [4/4] Re-evaluating challenge set on retrained model...")
    clf1 = TextClassifier(str(model_v1))
    acc1, details1 = evaluate_cases(clf1, URGENT_DRIFT_CASES)

    click.echo(f"        Challenge accuracy (retrained): {acc1:.1%}")
    _print_case_table(details1)

    delta = acc1 - acc0
    click.echo("")
    click.echo("  Summary")
    click.echo(f"    Baseline:  {acc0:.1%}")
    click.echo(f"    Retrained: {acc1:.1%}  ({delta:+.1%})")
    click.echo(f"    Log file:  {log_file}")
    if acc1 > acc0:
        click.echo("    ✓ Retrain improved accuracy on the challenge set.")
    else:
        click.echo("    ⚠ Accuracy did not improve — try more --retrain-epochs.")
    click.echo("")
