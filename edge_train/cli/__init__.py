"""CLI entry point."""

import click

from edge_train.cli.init import init
from edge_train.cli.train import train
from edge_train.cli.validate import validate
from edge_train.cli.deploy import deploy
from edge_train.cli.monitor import monitor
from edge_train.cli.cost import cost
from edge_train.cli.predict import predict


@click.group()
def main():
    """edge-train: TinyML continuous training agent.

    Train, deploy, and monitor tiny ML models on edge devices —
    all from the command line, zero GPU needed.
    """


main.add_command(init)
main.add_command(train)
main.add_command(validate)
main.add_command(deploy)
main.add_command(monitor)
main.add_command(cost)
main.add_command(predict)

if __name__ == "__main__":
    main()
