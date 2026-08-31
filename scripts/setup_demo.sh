#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQUIREMENTS_FILE="${PROJECT_ROOT}/773086/requirements.txt"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python was not found. Install Python 3.10-3.12 and retry." >&2
  exit 1
}
command -v node >/dev/null 2>&1 || {
  echo "Node.js was not found. Install Node.js 22.13 or newer and retry." >&2
  exit 1
}
command -v corepack >/dev/null 2>&1 || {
  echo "Corepack was not found. Install a Node.js distribution that includes Corepack." >&2
  exit 1
}
[[ -f "${REQUIREMENTS_FILE}" ]] || {
  echo "Python requirements file is missing: ${REQUIREMENTS_FILE}" >&2
  exit 1
}

"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Python 3.10-3.12 is required; found {sys.version.split()[0]}")
PY

node -e "const [major, minor] = process.versions.node.split('.').map(Number); if (major < 22 || (major === 22 && minor < 13)) { console.error('Node.js 22.13 or newer is required; found ' + process.versions.node); process.exit(1); }"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"

(
  cd "${PROJECT_ROOT}/frontend"
  corepack pnpm install --frozen-lockfile
)

if [[ "${SKIP_MODEL_PREFETCH:-0}" != "1" ]]; then
  echo "Prefetching and validating the frozen DINOv2 and SigLIP backbones..."
  (
    cd "${PROJECT_ROOT}/773086"
    PYTHONPATH=src AIGC_DEVICE=cpu "${VENV_DIR}/bin/python" - <<'PY'
from aigc_detector.api import LocalModelRuntime, RuntimeSettings
runtime = LocalModelRuntime(RuntimeSettings.from_environment())
runtime.ensure_loaded()
info = runtime.model_info()
runtime.shutdown()
print(f"Model cache ready: {info['forensic_backbone']} + {info['semantic_backbone']}")
PY
  )
else
  echo "Skipped model prefetch. The first detection will download the frozen backbones."
fi

echo
echo "Setup complete. Start the demo with: ./scripts/run_demo.sh"
