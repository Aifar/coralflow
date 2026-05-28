"""Focused project context for agent resume (not full chat history)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from edge_train.agent import AgentState


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_agent_context(
    state: AgentState, *, save: bool = True, **fields
) -> AgentState:
    """Merge known AgentState fields and persist."""
    allowed = AgentState.__dataclass_fields__
    for key, value in fields.items():
        if key in allowed and value is not None:
            setattr(state, key, value)
    state.updated_at = _now_iso()
    if save:
        state.save()
    return state


def _prediction_log_stats(log_path: str | Path) -> tuple[str, bool]:
    """Return (data_collection summary, needs_retrain hint)."""
    path = Path(log_path)
    if not path.exists():
        return "无预测采集记录", False

    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return "无预测采集记录", False

    labeled = sum(1 for e in entries if e.get("ground_truth"))
    unlabeled = len(entries) - labeled
    needs = labeled >= 10 and unlabeled == 0
    summary = f"{len(entries)} 条预测日志，{labeled} 条已标注"
    if unlabeled:
        summary += f"，{unlabeled} 条待标注"
    return summary, needs


def sync_agent_context(state: AgentState, *, save: bool = True) -> AgentState:
    """Refresh agent state from training history, deployments, and prediction log."""
    from edge_train.config import load_config
    from edge_train.deployments import DeploymentRegistry
    from edge_train.training_history import TrainingHistory

    history = TrainingHistory.load()
    history.sync_cloud_jobs()

    record = next(
        (r for r in history.records if r.status in ("running", "timeout")),
        None,
    )
    if record is None:
        record = next((r for r in history.records if r.status == "succeeded"), None)

    if record:
        purpose = (record.purpose or record.dataset_label or "").strip()
        if purpose:
            state.training_purpose = purpose
        if record.dataset_path:
            state.dataset_path = record.dataset_path
        if record.modality:
            state.modality = record.modality
        state.training_status = record.status
        if record.model_path:
            state.model_path = record.model_path
        state.task_type = record.modality or state.task_type

    dep = DeploymentRegistry.load().latest_vertex(
        state.model_path if state.model_path else None
    )
    if dep is None:
        dep = DeploymentRegistry.load().latest_vertex()
    if dep and dep.endpoint_name:
        state.deployment_status = "deployed"
        state.endpoint_name = dep.endpoint_name
        state.deployment_target = dep.endpoint_name
        if dep.model_path and not state.model_path:
            state.model_path = dep.model_path
    elif not state.deployment_status:
        state.deployment_status = "not_deployed"

    _, _, train_cfg, _ = load_config()
    data_collection, needs_retrain = _prediction_log_stats(
        train_cfg.prediction_log_path
    )
    state.data_collection = data_collection
    state.needs_retrain = needs_retrain
    state.updated_at = _now_iso()
    if save:
        state.save()
    return state


def format_project_context(state: AgentState) -> str:
    """Compact resume block for the LLM — training/deployment/data status only."""
    lines: list[str] = []
    if state.training_purpose:
        lines.append(f"训练目的/项目名: {state.training_purpose}")
    if state.dataset_path:
        lines.append(f"数据集: {state.dataset_path}")
    if state.modality:
        lines.append(f"模态: {state.modality}")
    if state.training_status:
        lines.append(f"训练状态: {state.training_status}")
    if state.model_path:
        lines.append(f"模型: {state.model_path}")
    if state.deployment_status:
        dep = state.deployment_status
        if state.endpoint_name:
            dep = f"{dep} → {state.endpoint_name}"
        lines.append(f"部署: {dep}")
    if state.data_collection:
        lines.append(f"数据采集: {state.data_collection}")
    if state.needs_retrain:
        lines.append("需要重新训练: 是（已有足够标注反馈）")
    elif state.data_collection and state.data_collection != "无预测采集记录":
        lines.append("需要重新训练: 否（或待更多标注）")
    if state.last_step:
        lines.append(f"上次步骤: {state.last_step}")
    return "\n".join(lines)
