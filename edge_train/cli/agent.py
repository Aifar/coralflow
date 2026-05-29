"""agent command — LLM-powered interactive agent for CoralFlow."""

import click


@click.command()
@click.option("--api-key", "-k", default=None, help="LLM API key override")
@click.option("--model", "-m", default=None, help="LLM model override")
@click.option("--endpoint", default=None, help="LLM API endpoint override")
@click.option("--resume", is_flag=True, help="Skip scan, resume from last state")
def agent(
    api_key: str | None,
    model: str | None,
    endpoint: str | None,
    resume: bool,
):
    """Start the interactive LLM-powered CoralFlow agent.

    The agent discovers datasets, assesses local resources, validates data,
    trains models, and deploys — all through natural conversation.

    Requires a working LLM API key (environment, .env, or --api-key).
    """
    from edge_train.config import load_config
    from edge_train.agent.llm import LLMConfig, ensure_llm_client
    from edge_train.agent.loop import run_agent_loop
    from edge_train.agent.ui import CoralFlowUI
    from edge_train.agent import AgentState, DatasetScanner, scan_models

    load_config()  # load .env (GCP + LLM keys) before REPL / subprocess tools

    ui = CoralFlowUI()

    # Load config
    config = LLMConfig.from_env()

    if api_key:
        config.api_key = api_key
    if model:
        config.model = model
    if endpoint:
        config.endpoint = endpoint

    def _echo(msg: str) -> None:
        ui.error(msg.strip())

    llm = ensure_llm_client(
        config,
        _llm_prompt_fn(ui),
        echo=_echo,
    )

    if api_key or model or endpoint:
        from edge_train.agent.llm import persist_llm_config

        persist_llm_config(llm.config)

    from edge_train.agent.google_env import ensure_google_env_at_startup

    ensure_google_env_at_startup(_llm_prompt_fn(ui), echo=_echo)

    # Load agent state
    state = AgentState.load()

    # Build context for the loop banner
    ctx_parts = [f"LLM: {llm.config.model}"]
    if llm.config.endpoint != "https://api.openai.com/v1":
        ctx_parts.append(f"Endpoint: {llm.config.endpoint}")

    if not resume:
        datasets = DatasetScanner.scan()
        models = scan_models()
        ctx_parts.append(f"Datasets: {len(datasets)} found")
        ctx_parts.append(f"Models: {len(models)} found")

        scan_lines = []
        if datasets:
            scan_lines.append(f"Found {len(datasets)} dataset(s):")
            for d in datasets:
                classes_str = ", ".join(d.get("classes", [])[:5])
                if len(d.get("classes", [])) > 5:
                    classes_str += f" (+{len(d['classes']) - 5} more)"
                scan_lines.append(
                    f"  • {d['name']} — {d['rows']} rows, {d.get('modality', '?')}, "
                    f"[{classes_str}] ({d['source']})"
                )
        if models:
            scan_lines.append(f"Found {len(models)} model(s):")
            for m in models:
                classes_str = ", ".join(m.get("classes", [])[:5])
                scan_lines.append(f"  • {m['name']} — [{classes_str}]")

        from edge_train.training_history import TrainingHistory, format_startup_summary

        history = TrainingHistory.load()
        history.sync_cloud_jobs()
        training_summary = format_startup_summary(history)
        if training_summary:
            scan_lines.append(training_summary)

        scan_result = (
            "\n".join(scan_lines) if scan_lines else "No datasets or models found."
        )
    else:
        scan_result = ""
        ctx_parts.append("Resumed from previous session")

    if state.dataset_path:
        ctx_parts.append(f"Last dataset: {state.dataset_path}")

    ctx_summary = " | ".join(ctx_parts)

    try:
        run_agent_loop(llm, state, scan_result, ctx_summary, llm_enabled=True)
    except KeyboardInterrupt:
        click.echo()


def _llm_prompt_fn(ui):
    def _prompt(label: str, *, default: str = "") -> str:
        if default:
            ui.info(f"Default: {default}")
        try:
            return ui.prompt(label)
        except (EOFError, KeyboardInterrupt):
            return "skip" if "API key" in label else default

    return _prompt
