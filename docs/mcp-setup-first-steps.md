# MCP server — first steps

Follow these in order so **strain-tools** runs locally (Cursor) and optionally on the network (Prompt Opinion + ngrok).

## Step 1 — Python environment and package

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Step 2 — Data and models the tools expect

- **Kaggle / CSV tools:** place [`data/emotions.csv`](../data/emotions.csv) (or set `STRAIN_EMOTIONS_CSV`) and train the classifier:

  ```bash
  python scripts/train_baseline.py
  ```

- **DREAMER tools (optional):** export epochs and train VAD (see root [README.md](../README.md)). If you skip this, DREAMER-only tools will error until manifests and `dreamer_vad_multiridge.joblib` exist.

## Step 3 — Run MCP locally (stdio)

Default transport is **stdio** (good for Cursor and Claude Desktop):

```bash
source .venv/bin/activate
python -m mcp_server.server
```

Or use the helper script:

```bash
chmod +x scripts/run_mcp_stdio.sh
./scripts/run_mcp_stdio.sh
```

The process stays in the foreground; the **client** (Cursor) spawns this process and talks over stdin/stdout.

## Step 4 — Wire Cursor to this repo

**Option A — Project config (recommended):** this repo includes [`.cursor/mcp.json`](../.cursor/mcp.json). Cursor loads MCP servers from there when the folder is open. The `command` uses `${workspaceFolder}/.venv/bin/python`; if that does not resolve in your Cursor build, replace it with the **absolute** path to `.venv/bin/python` in this repo.

**Option B — User settings:** open **Cursor Settings → MCP → Add server**, and use:

| Field | Value |
|--------|--------|
| **Name** | `strain-tools` |
| **Command** | Full path to `.venv/bin/python` in this repo |
| **Args** | `-m`, `mcp_server.server` |
| **Cwd** | Repository root |

Set env `STRAIN_MCP_TRANSPORT=stdio` if you ever changed the default elsewhere.

Restart Cursor or reload MCP, then open the MCP panel and confirm **strain-tools** lists tools (e.g. `load_dataset_tool`).

## Step 5 — Optional: SSE + ngrok (Prompt Opinion)

1. Copy [`.env.example`](../.env.example) to `.env` and tune `FASTMCP_HOST` / `FASTMCP_PORT`, or export vars in the shell.
2. Run:

   ```bash
   chmod +x scripts/run_mcp_sse.sh
   ./scripts/run_mcp_sse.sh
   ```

3. In another terminal: `ngrok http 8765` (or your `FASTMCP_PORT`).
4. Use the HTTPS forwarding URL + **SSE path** (default **`/sse`**) when the Prompt Opinion workspace asks for the MCP server URL, e.g. `https://xxxx.ngrok-free.app/sse`.
5. Keep **`STRAIN_MCP_RELAX_DNS=1`** behind ngrok so the tunneled `Host` header is accepted (development only).

More context: [prompt-opinion-hackathon.md](./prompt-opinion-hackathon.md).

## Troubleshooting

| Issue | What to try |
|--------|--------------|
| Cursor never lists tools | Confirm `python -m mcp_server.server` uses the same venv where `pip install -e .` was run; check **MCP logs** in Cursor. |
| `FileNotFoundError` on tools | `data/emotions.csv` and/or `strain/models/baseline_pipeline.joblib` missing — complete Step 2. |
| SSE / ngrok 403 or connection closed | Set `STRAIN_MCP_RELAX_DNS=1`; confirm ngrok URL includes the correct path (`/sse`). |
| Port in use | Change `FASTMCP_PORT` and restart. |
