# Prompt Opinion — Agents Assemble hackathon alignment

**First steps to run the MCP server locally or with SSE:** [mcp-setup-first-steps.md](./mcp-setup-first-steps.md).

This note maps the **“Agents Assemble – The Healthcare AI Endgame Challenge”** workflow (Prompt Opinion + MCP + A2A + FHIR) to this repo and the official GitHub ecosystem.

## Official entry points

| Resource | URL |
|----------|-----|
| Prompt Opinion (marketing / challenge) | [https://www.promptopinion.ai/](https://www.promptopinion.ai/) |
| Workspace / builder (per hackathon brief) | [https://app.promptopinion.ai/](https://app.promptopinion.ai/) |
| SHARP-on-MCP spec (FHIR-oriented MCP tools) | [https://www.sharponmcp.com/](https://www.sharponmcp.com/) |
| Platform docs (referenced from `po-overview`) | [https://docs.promptopinion.ai/](https://docs.promptopinion.ai/) |

## GitHub org **[@prompt-opinion](https://github.com/prompt-opinion)** — what each repo is for

The brief’s **“po-sdk”** maps to the **Google ADK sample agents**, not a repo literally named `po-sdk`:

| Repository | Role |
|------------|------|
| [**po-overview**](https://github.com/prompt-opinion/po-overview) | Ecosystem hub: standards (MCP, A2A, FHIR), **repository map**, how builders vs organizations connect. |
| [**po-community-mcp**](https://github.com/prompt-opinion/po-community-mcp) | **Community MCP server** pattern — extra **FHIR-related tools** on top of the default SHARP-on-MCP style server used with Prompt Opinion. |
| [**po-adk-python**](https://github.com/prompt-opinion/po-adk-python) | **Python A2A agents** (Google ADK): `healthcare_agent`, `general_agent`, `orchestrator`; **A2A v1** agent cards; **FHIR context** in metadata; **deploy + register** with Prompt Opinion. |
| [**po-adk-typescript**](https://github.com/prompt-opinion/po-adk-typescript) | Same idea in TypeScript. |

**EmotiScan’s `mcp_server`** is a separate **Python FastMCP** tool server (EEG / emotion / DREAMER). It complements (does not replace) `po-community-mcp` for FHIR-heavy tools. Long term you can:

- **Reuse patterns** from `po-community-mcp` / SHARP-on-MCP for FHIR exports, or  
- **Call** this repo’s FastAPI from a **po-adk-python**-style agent as custom tools.

## Hackathon paths vs this repo

### Option 1 — No-code agent (Prompt Opinion UI)

- Sign in at **app.promptopinion.ai**, connect **Gemini** (e.g. **Gemini 1.5 Flash**).
- Grounding: synthetic FHIR bundle import, upload FHIR JSON, or manual patient + notes.
- Enable **A2A** and **FHIR context** in workspace settings per brief.
- Use Launchpad to chat and validate patient-aware behavior.

**EmotiScan:** use for narrative + demo data; wire **FHIR `DiagnosticReport` / `Observation`** export (planned in product spec) to match this path.

### Option 2 — **Custom MCP server** (best fit for EmotiScan tools today)

Prompt Opinion expects a **reachable MCP endpoint** (the brief mentions **ngrok**).

1. **Export DREAMER epochs** and optional VAD model (see root `README.md`).
2. Run this repo’s MCP server over **SSE** (not stdio), bind publicly or behind ngrok:

```bash
cd /path/to/strain
source .venv/bin/activate
export EMOTISCAN_MCP_TRANSPORT=sse
export FASTMCP_HOST=0.0.0.0
export FASTMCP_PORT=8765
# Allow tunneled Host headers (ngrok / Cloud Run preview):
export EMOTISCAN_MCP_RELAX_DNS=1
python -m mcp_server.server
```

3. Start **ngrok** (or Cloudflare Tunnel, etc.) on the same port, e.g. `ngrok http 8765`.
4. In Prompt Opinion workspace, add the MCP server URL your platform expects — typically the **SSE base URL** from FastMCP (defaults include paths such as `/sse`; check `FASTMCP_SSE_PATH` / `FASTMCP_MESSAGE_PATH` via env or [FastMCP HTTP deployment](https://gofastmcp.com/deployment/http)).
5. **Marketplace Studio:** publish the agent or MCP server before judging, per brief.

**Tools exposed today:** `load_dataset_tool`, CSV feature/classify/explain/screen tools, DREAMER epoch feature + VAD tools (see `mcp_server/server.py`).

### Option 3 — **Custom A2A agent** (advanced)

- Follow [**po-adk-python**](https://github.com/prompt-opinion/po-adk-python) README section **“Connecting to Prompt Opinion”**: deploy a **public URL**, set `HEALTHCARE_AGENT_URL` (or your agent’s URL), `PO_PLATFORM_BASE_URL`, register **`.well-known/agent-card.json`**, use **`X-API-Key`** from the workspace.
- Prompt Opinion injects **FHIR server URL**, **bearer token**, **patient ID** into A2A metadata; tools read from session state — ideal for a **clinical wrapper** around EmotiScan.

**EmotiScan:** implement an ADK agent that delegates “emotion / stress screening from EEG summary” to your **FastAPI** (`/analyze`, `/analyze/dreamer`) or in-process Python, while using PO’s FHIR tools for demographics and reporting.

## Standards stack (from `po-overview`)

- **MCP** — tool discovery and invocation (this repo’s `emotiscan-tools` server).  
- **A2A** — multi-agent coordination (`po-adk-python` / `a2aproject/a2a-python`).  
- **FHIR** — patient context and interoperable outputs ([community MCP](https://github.com/prompt-opinion/po-community-mcp), SHARP-on-MCP).

## Suggested build order for the challenge

1. **MCP on SSE + ngrok** → register in Prompt Opinion → demonstrate tools in Launchpad.  
2. **Synthetic FHIR patient** in workspace + align copy with “not a medical device”.  
3. **`export_fhir`-style bundle** from screening results (extend API/MCP).  
4. Optional: **fork `po-adk-python`** and add a thin **EmotiScan specialist** agent with A2A v1 card.

## Community

- **Discord** — hackathon / office hours (link from Prompt Opinion site or challenge page).  
- Issues on **po-overview** for cross-repo platform questions.
