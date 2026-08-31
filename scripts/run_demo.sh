#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
DEVICE="${AIGC_DEVICE:-auto}"
API_URL="${NEXT_PUBLIC_API_BASE_URL:-http://${BACKEND_HOST}:${BACKEND_PORT}}"

[[ -x "${VENV_DIR}/bin/python" ]] || {
  echo "Python environment is missing. Run ./scripts/setup_demo.sh first." >&2
  exit 1
}
[[ -d "${PROJECT_ROOT}/frontend/node_modules" ]] || {
  echo "Frontend dependencies are missing. Run ./scripts/setup_demo.sh first." >&2
  exit 1
}

backend_pid=""
frontend_pid=""
cleanup() {
  trap - EXIT INT TERM
  [[ -n "${frontend_pid}" ]] && kill "${frontend_pid}" 2>/dev/null || true
  [[ -n "${backend_pid}" ]] && kill "${backend_pid}" 2>/dev/null || true
  wait "${frontend_pid}" 2>/dev/null || true
  wait "${backend_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(
  cd "${PROJECT_ROOT}/773086"
  PYTHONPATH=src "${VENV_DIR}/bin/python" scripts/serve_demo.py \
    --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" --device "${DEVICE}"
) &
backend_pid=$!

for _attempt in $(seq 1 60); do
  if curl --fail --silent "http://${BACKEND_HOST}:${BACKEND_PORT}/api/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    echo "The inference backend stopped during startup." >&2
    exit 1
  fi
  sleep 0.5
done

curl --fail --silent "http://${BACKEND_HOST}:${BACKEND_PORT}/api/health" >/dev/null || {
  echo "The inference backend did not become healthy on port ${BACKEND_PORT}." >&2
  exit 1
}

(
  cd "${PROJECT_ROOT}/frontend"
  NEXT_PUBLIC_API_BASE_URL="${API_URL}" corepack pnpm exec vinext dev \
    --hostname "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
) &
frontend_pid=$!

echo
echo "RobustFusion is starting:"
echo "  Demo:    http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "  API:     http://${BACKEND_HOST}:${BACKEND_PORT}/docs"
echo "  Device:  ${DEVICE}"
echo
echo "Press Ctrl+C to stop both processes."

while kill -0 "${backend_pid}" 2>/dev/null && kill -0 "${frontend_pid}" 2>/dev/null; do
  sleep 1
done

echo "One of the demo services stopped unexpectedly." >&2
exit 1
