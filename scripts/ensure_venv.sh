#!/usr/bin/env bash
# Create and activate .venv if not already in a virtual environment.
# Usage:
#   source scripts/ensure_venv.sh
#   ./scripts/ensure_venv.sh pip install -e ".[dev]"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${CORALFLOW_VENV:-$ROOT/.venv}"

_venv_python() {
  if [[ -x "$VENV/bin/python" ]]; then
    echo "$VENV/bin/python"
  else
    echo ""
  fi
}

_create_venv() {
  echo "Creating virtual environment at $VENV ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$VENV"
  elif command -v python >/dev/null 2>&1; then
    python -m venv "$VENV"
  else
    echo "Error: python3 not found; cannot create virtual environment." >&2
    return 1
  fi
}

_venv_abs() {
  mkdir -p "$VENV"
  (cd "$VENV" && pwd)
}

_ensure_venv_dir() {
  local target current
  target="$(_venv_abs)"
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    current="$(cd "$VIRTUAL_ENV" && pwd)"
    if [[ "$current" == "$target" ]]; then
      return 0
    fi
  fi
  if [[ ! -x "$target/bin/python" ]]; then
    _create_venv
  fi
  if [[ ! -f "$target/bin/activate" ]]; then
    echo "Error: $target exists but is not a valid venv (missing bin/activate)." >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "$target/bin/activate"
}

_ensure_venv_dir

PY="$(_venv_python)"
if [[ -n "$PY" ]] && ! "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "pip" ]]; then
    shift
    set -- python -m pip "$@"
  fi
  exec "$@"
fi
