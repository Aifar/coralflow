"""agent command — LLM-powered interactive agent for CoralFlow."""

import sys

import click


@click.command()
@click.option("--model", "-m", default=None, help="LLM model override")
@click.option("--endpoint", default=None, help="LLM API endpoint override")
@click.option("--resume", is_flag=True, help="Skip scan, resume from last state")
def agent(model: str | None, endpoint: str | None, resume: bool):
    """Start the interactive LLM-powered CoralFlow agent.

    The agent discovers datasets, assesses local resources, validates data,
    trains models, and deploys — all through natural conversation.

    Requires CORALFLOW_LLM_API_KEY to be set.
    """
    from edge_train.config import load_config
    from edge_train.agent.llm import LLMClient, LLMConfig
    from edge_train.agent.loop import run_agent_loop
    from edge_train.agent import AgentState, DatasetScanner, scan_models

    load_config()  # load .env (GCP + LLM keys) before REPL / subprocess tools

    # Load config
    config = LLMConfig.from_env()

    if model:
        config.model = model
    if endpoint:
        config.endpoint = endpoint

    if not config.is_valid():
        click.echo(
            "Error: CORALFLOW_LLM_API_KEY is not set.\n"
            "Set it in your environment or .env file:\n"
            "  export CORALFLOW_LLM_API_KEY=sk-...\n"
            "\n"
            "Optional:\n"
            "  CORALFLOW_LLM_ENDPOINT  (default: https://api.openai.com/v1)\n"
            "  CORALFLOW_LLM_MODEL      (default: gpt-4o)",
            err=True,
        )
        sys.exit(1)

    # Load agent state and sync from training history / deployments
    state = AgentState.load()
    from edge_train.agent.context import format_project_context, sync_agent_context

    state = sync_agent_context(state)

    # Build context for the loop banner
    ctx_parts = [f"LLM: {config.model}"]
    if config.endpoint != "https://api.openai.com/v1":
        ctx_parts.append(f"Endpoint: {config.endpoint}")

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

    project_ctx = format_project_context(state)
    if project_ctx:
        ctx_parts.append("Project context loaded")

    if state.dataset_path:
        ctx_parts.append(f"Dataset: {state.dataset_path}")
    if state.training_purpose:
        ctx_parts.append(f"Project: {state.training_purpose}")

    ctx_summary = " | ".join(ctx_parts)

    # Create LLM client and enter REPL
    llm = LLMClient(config)

    try:
        run_agent_loop(llm, state, scan_result, ctx_summary)
    except KeyboardInterrupt:
        click.echo()
