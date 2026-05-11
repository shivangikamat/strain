#!/usr/bin/env bash
# Run STRAIN MCP over SSE on 0.0.0.0:8765 — tunnel with ngrok, then register the public /sse URL in Prompt Opinion.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck source=/dev/null
  set -o allexport; source "$ROOT/.env"; set +o allexport
fi
export STRAIN_MCP_TRANSPORT=${STRAIN_MCP_TRANSPORT:-sse}
export FASTMCP_HOST=${FASTMCP_HOST:-0.0.0.0}
export FASTMCP_PORT=${FASTMCP_PORT:-8765}
export STRAIN_MCP_RELAX_DNS=${STRAIN_MCP_RELAX_DNS:-1}
echo "MCP SSE on http://${FASTMCP_HOST}:${FASTMCP_PORT} (see FASTMCP_SSE_PATH, default /sse)"
echo "Example ngrok: ngrok http ${FASTMCP_PORT}"
exec python -m mcp_server.server
