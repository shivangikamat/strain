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

- **Kaggle / CSV tools:** place `[data/emotions.csv](../data/emotions.csv)` (or set `STRAIN_EMOTIONS_CSV`) and train the classifier:
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

**Option A — Project config (recommended):** this repo includes `[.cursor/mcp.json](../.cursor/mcp.json)`. Cursor loads MCP servers from there when the folder is open. The `command` uses `${workspaceFolder}/.venv/bin/python`; if that does not resolve in your Cursor build, replace it with the **absolute** path to `.venv/bin/python` in this repo.

**Option B — User settings:** open **Cursor Settings → MCP → Add server**, and use:


| Field       | Value                                        |
| ----------- | -------------------------------------------- |
| **Name**    | `strain-tools`                               |
| **Command** | Full path to `.venv/bin/python` in this repo |
| **Args**    | `-m`, `mcp_server.server`                    |
| **Cwd**     | Repository root                              |


Set env `STRAIN_MCP_TRANSPORT=stdio` if you ever changed the default elsewhere.

Restart Cursor or reload MCP, then open the MCP panel and confirm **strain-tools** lists tools (e.g. `load_dataset_tool`).

## Step 5 — Optional: HTTP + ngrok (Prompt Opinion)

Per official docs: register remote MCP under `**Configuration → MCP Servers`** ([FHIR Context With MCP](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context)), then attach that server to a **BYO agent** on the **Tools** tab ([BYO Agents](https://docs.promptopinion.ai/agents/byo-agents)).

**Important — transport must match what Prompt Opinion sends:**

| Po **Transport** dropdown | Run this locally | Endpoint URL (after ngrok HTTPS host) |
| ------------------------- | ---------------- | --------------------------------------- |
| **SSE** (or “HTTP + SSE”) | `./scripts/run_mcp_sse.sh` | `https://xxxx.ngrok-free.app/sse` |
| **Streamable HTTP** | `./scripts/run_mcp_streamable_http.sh` | `https://xxxx.ngrok-free.app/mcp` (default; override with `FASTMCP_STREAMABLE_HTTP_PATH`) |

If Po is **Streamable HTTP** but the URL ends in **`/sse`**, Po will **POST** to `/sse` and the server will respond **405 Method Not Allowed** (SSE expects **GET** on `/sse`).

1. Copy `[.env.example](../.env.example)` to `.env` and tune `FASTMCP_HOST` / `FASTMCP_PORT`, or export vars in the shell.
2. Run **one** of:
  ```bash
   chmod +x scripts/run_mcp_sse.sh scripts/run_mcp_streamable_http.sh
   ./scripts/run_mcp_sse.sh              # Po transport: SSE → …/sse
   # or
   ./scripts/run_mcp_streamable_http.sh  # Po transport: Streamable HTTP → …/mcp
   ```
3. In another terminal: `ngrok http 8765` (or your `FASTMCP_PORT`).
4. In Po `**Configuration → MCP Servers**`: set **Transport** and **Endpoint** as in the table above, then **Continue** (Po sends `initialize`).
5. Keep `**STRAIN_MCP_RELAX_DNS=1**` behind ngrok so the tunneled `Host` header is accepted (development only).
6. `**Agents → BYO Agents**`: edit your agent → **Tools** → select **strain-tools**.

Full Po checklist + FHIR extension gap: [hackathon-remaining-tasks.md](./hackathon-remaining-tasks.md) · [prompt-opinion-hackathon.md](./prompt-opinion-hackathon.md).

## Troubleshooting


| Issue                                | What to try                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Cursor never lists tools             | Confirm `python -m mcp_server.server` uses the same venv where `pip install -e .` was run; check **MCP logs** in Cursor. |
| `FileNotFoundError` on tools         | `data/emotions.csv` and/or `strain/models/baseline_pipeline.joblib` missing — complete Step 2.                           |
| SSE / ngrok 403 or connection closed | Set `STRAIN_MCP_RELAX_DNS=1`; confirm ngrok URL includes the correct path (`/sse`).                                      |
| ngrok shows **POST /sse → 405** | Po **Transport** is **Streamable HTTP** but the URL is **`/sse`**. Either switch Po to **SSE** + keep `…/sse`, or run `./scripts/run_mcp_streamable_http.sh` and set the endpoint to **`…/mcp`**. |
| ngrok **ERR_NGROK_8012** (upstream `localhost:8765` failed) | Start `./scripts/run_mcp_sse.sh` **first** and leave it running; `curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:$FASTMCP_PORT/sse` should not refuse the connection. Use the **same** port in ngrok as `FASTMCP_PORT` (default **8765** in the script). |
| Port in use                          | Change `FASTMCP_PORT` and restart.                                                                                       |
| `ModuleNotFoundError: No module named 'fhir'` | Run **`pip install -e .`** from the repo root so **`fhir.resources`** (in `pyproject.toml`) is installed in this venv.   |


