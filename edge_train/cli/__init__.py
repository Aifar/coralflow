"""CLI entry point."""

import click

import edge_train.config  # noqa: F401 — bootstrap .env before subcommands

from edge_train.cli.init import init
from edge_train.cli.train import train
from edge_train.cli.validate import validate
from edge_train.cli.deploy import deploy
from edge_train.cli.monitor import monitor
from edge_train.cli.cost import cost
from edge_train.cli.predict import predict
from edge_train.cli.simulate import simulate
from edge_train.cli.agent import agent
from edge_train.cli.demo import demo
from edge_train.cli.models import models


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """coralflow: TinyML continuous training agent.

    Train, deploy, and monitor tiny ML models on edge devices —
    all from the command line, zero GPU needed.

    Run without a subcommand to start the interactive agent (same as `agent`).
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(agent)


main.add_command(agent)
main.add_command(init)
main.add_command(train)
main.add_command(validate)
main.add_command(deploy)
main.add_command(monitor)
main.add_command(cost)
main.add_command(predict)
main.add_command(simulate)
main.add_command(demo)
main.add_command(models)

if __name__ == "__main__":
    main()
