#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

[[ -x "${VENV_DIR}/bin/python" ]] || {
  echo "Run ./scripts/setup_demo.sh before checks." >&2
  exit 1
}

(
  cd "${PROJECT_ROOT}/773086"
  PYTHONPATH=src "${VENV_DIR}/bin/python" -m unittest discover -s tests -p 'test_api.py'
)
(
  cd "${PROJECT_ROOT}/frontend"
  corepack pnpm lint
  corepack pnpm build
)

echo "Backend contract tests, frontend lint, and production build passed."
