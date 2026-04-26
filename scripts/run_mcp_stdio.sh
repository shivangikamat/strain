#!/usr/bin/env bash
# Run EmotiScan MCP over stdio (default) — for Cursor / Claude Desktop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi
export EMOTISCAN_MCP_TRANSPORT=stdio
exec python -m mcp_server.server
