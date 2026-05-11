#!/usr/bin/env bash
# Run STRAIN MCP over Streamable HTTP (default FastMCP path /mcp) — use when Prompt Opinion
# transport is "Streamable HTTP", not SSE.
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
# Hard-override transport settings — .env may set stdio/localhost for local use
export STRAIN_MCP_TRANSPORT=streamable-http
export FASTMCP_HOST=0.0.0.0
export FASTMCP_PORT=${FASTMCP_PORT:-8765}
export STRAIN_MCP_RELAX_DNS=1
echo "MCP Streamable HTTP on http://${FASTMCP_HOST}:${FASTMCP_PORT}${FASTMCP_STREAMABLE_HTTP_PATH:-/mcp}"
echo "Example ngrok: ngrok http ${FASTMCP_PORT}"
echo "Prompt Opinion endpoint: https://<subdomain>.ngrok-free.app${FASTMCP_STREAMABLE_HTTP_PATH:-/mcp} (transport: Streamable HTTP)"
exec python -m mcp_server.server
