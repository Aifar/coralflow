"""Terminal UI — rich rendering + prompt_toolkit input for coralflow agent."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


class CoralFlowUI:
    """Rich rendering + prompt_toolkit input for the coralflow agent REPL."""

    def __init__(self) -> None:
        self.console = Console(highlight=False)
        self._session: Any = None
        self._has_tty = sys.stdout.isatty() and sys.stdin.isatty()

    @property
    def session(self) -> Any:
        if self._session is None and self._has_tty:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.styles import Style

            self._session = PromptSession(
                bottom_toolbar=" /help | /exit | Ctrl+C to quit ",
                style=Style.from_dict(
                    {
                        "prompt": "bold cyan",
                        "bottom-toolbar": "bg:#2d2d2d #888888",
                    }
                ),
            )
        return self._session

    # ── Rendering ─────────────────────────────────────────────────────────

    def markdown(self, text: str) -> None:
        """Render text as Markdown via rich."""
        if text:
            self.console.print(Markdown(text))

    def separator(self) -> None:
        """Print a full-width horizontal rule."""
        self.console.print(Rule(style="cyan"))

    def step(self, title: str) -> None:
        """Print a styled section header rule."""
        self.console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))

    def panel(self, text: str, title: str | None = None) -> None:
        """Render text inside a bordered panel."""
        self.console.print(Panel(Markdown(text), title=title, border_style="cyan"))

    def info(self, text: str) -> None:
        """Dimmed / italic status message."""
        self.console.print(f"[dim italic]{text}[/dim italic]")

    def notice(self, text: str) -> None:
        """Neutral guidance text for setup prompts and configuration menus."""
        if text:
            self.console.print(text)

    def error(self, text: str) -> None:
        """Red bold error message."""
        self.console.print(f"[bold red]{text}[/bold red]")

    def tool_start(self, name: str, detail: str | None = None) -> None:
        """Tool execution indicator."""
        if detail:
            self.console.print(
                f"[bold blue]  ⚙ {name}:[/bold blue] [cyan]{detail}[/cyan]"
            )
        else:
            self.console.print(f"[bold blue]  ⚙ {name}...[/bold blue]")

    def success(self, text: str) -> None:
        """Green checkmark + message."""
        self.console.print(f"[bold green]✓ {text}[/bold green]")

    def raw(self, text: str) -> None:
        """Print plain text (for streaming / progress output)."""
        self.console.print(Text(text, end=""))

    # ── Input ─────────────────────────────────────────────────────────────

    def prompt(self, text: str) -> str:
        """Prompt for input with bottom toolbar.

        Falls back to plain input() when not attached to a TTY
        (e.g. inside a Click CliRunner test).
        """
        if self.session is not None:
            try:
                return self.session.prompt(f"{text}> ")
            except (EOFError, KeyboardInterrupt):
                raise
        # Fallback for non-TTY environments
        return input(f"{text}> ")

    def choose(self, options: list[str], allow_custom: bool = True) -> str | None:
        """Print numbered options and return the selected one.

        If the user types a number, returns the corresponding option.
        If allow_custom and user types 0, prompts for custom text.
        Otherwise, returns the typed text directly.
        """
        self.console.print()
        for i, opt in enumerate(options, 1):
            self.console.print(f"  {i}. {opt}")
        if allow_custom:
            self.console.print("  0. (type your own)")
        self.console.print()

        try:
            choice = self.prompt("choice")
        except (EOFError, KeyboardInterrupt):
            return None

        try:
            idx = int(choice)
            if allow_custom and idx == 0:
                try:
                    return self.prompt("custom")
                except (EOFError, KeyboardInterrupt):
                    return None
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            return choice

        return None

    def confirm(self, text: str) -> bool:
        """Prompt for a yes/no confirmation."""
        try:
            result = self.prompt(f"{text} [Y/n]")
            return result.lower() in ("y", "yes", "")
        except (EOFError, KeyboardInterrupt):
            return False
