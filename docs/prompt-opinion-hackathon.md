# Prompt Opinion — Agents Assemble hackathon alignment

**Run STRAIN MCP locally or over SSE:** [mcp-setup-first-steps.md](./mcp-setup-first-steps.md).

**Remaining STRAIN tasks vs spec / Po:** [hackathon-remaining-tasks.md](./hackathon-remaining-tasks.md).

## Official Prompt Opinion documentation

| Topic | URL |
|-------|-----|
| Getting Started (LLM key, register, patient, agent, conversation) | [docs.promptopinion.ai/getting-started](https://docs.promptopinion.ai/getting-started) |
| Model Configuration (`Configuration → Models`) | [docs.promptopinion.ai/model-configuration](https://docs.promptopinion.ai/model-configuration) |
| A2A **v1** migration (agent card: `supportedInterfaces`, no top-level `url`) | [docs.promptopinion.ai/a2a-v1-migration](https://docs.promptopinion.ai/a2a-v1-migration) |
| Agent Scopes (Workspace / Patient / Group) | [docs.promptopinion.ai/agents/agent-scopes](https://docs.promptopinion.ai/agents/agent-scopes) |
| BYO Agents (system prompt, **Tools = MCP servers**, A2A, FHIR) | [docs.promptopinion.ai/agents/byo-agents](https://docs.promptopinion.ai/agents/byo-agents) |
| External Agents (`/.well-known/agent-card.json`, consult flow) | [docs.promptopinion.ai/agents/external-agents](https://docs.promptopinion.ai/agents/external-agents) |
| FHIR context overview (URL, token, patient id) | [docs.promptopinion.ai/fhir-context/overview](https://docs.promptopinion.ai/fhir-context/overview) |
| **MCP + FHIR** (`ai.promptopinion/fhir-context`, headers on tool calls) | [docs.promptopinion.ai/fhir-context/mcp-fhir-context](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context) |
| **A2A + FHIR** (extension + message `metadata`) | [docs.promptopinion.ai/fhir-context/a2a-fhir-context](https://docs.promptopinion.ai/fhir-context/a2a-fhir-context) |

## Web app & GitHub

| Resource | URL |
|----------|-----|
| Prompt Opinion | [https://www.promptopinion.ai/](https://www.promptopinion.ai/) |
| Workspace | [https://app.promptopinion.ai/](https://app.promptopinion.ai/) |
| Docs home | [https://docs.promptopinion.ai/](https://docs.promptopinion.ai/) |
| SHARP-on-MCP | [https://www.sharponmcp.com/](https://www.sharponmcp.com/) |

## GitHub org [@prompt-opinion](https://github.com/prompt-opinion)

| Repository | Role |
|------------|------|
| [**po-overview**](https://github.com/prompt-opinion/po-overview) | Ecosystem hub, repo map. |
| [**po-community-mcp**](https://github.com/prompt-opinion/po-community-mcp) | Community MCP / FHIR-oriented patterns. |
| [**po-adk-python**](https://github.com/prompt-opinion/po-adk-python) | Python A2A agents (Google ADK), agent cards, Po registration. |
| [**po-adk-typescript**](https://github.com/prompt-opinion/po-adk-typescript) | Same in TypeScript. |

**STRAIN’s `strain-tools` MCP** (`mcp_server/`) is separate: EEG / CSV / DREAMER tools. Attach it to a **BYO agent** under **Tools** after registering it under **`Configuration → MCP Servers`** per Po docs.

---

## How to include STRAIN MCP in your Prompt Opinion app

1. Complete Po basics: [Getting Started](https://docs.promptopinion.ai/getting-started) + [Model Configuration](https://docs.promptopinion.ai/model-configuration).
2. Run STRAIN MCP over **SSE** with a **public HTTPS URL** (e.g. ngrok). See [mcp-setup-first-steps.md](./mcp-setup-first-steps.md) § Step 5 and commands below.
3. In Po: **`Configuration → MCP Servers`** → add your server URL. Po issues **`initialize`**; if you implement Po’s extension, declare **`ai.promptopinion/fhir-context`** per [FHIR Context With MCP](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context).
4. **`Agents → BYO Agents`** → create/edit agent → **Tools** tab → select this MCP server ([BYO Agents](https://docs.promptopinion.ai/agents/byo-agents)).
5. Test on **Launchpad** with the right **scope** ([Agent Scopes](https://docs.promptopinion.ai/agents/agent-scopes)).

### SSE commands (reference)

```bash
cd /path/to/strain
source .venv/bin/activate
export STRAIN_MCP_TRANSPORT=sse
export FASTMCP_HOST=0.0.0.0
export FASTMCP_PORT=8765
export STRAIN_MCP_RELAX_DNS=1   # needed behind ngrok / odd Host headers
python -m mcp_server.server
```

Then e.g. `ngrok http 8765` and register **`https://<host>/sse`** (confirm path with FastMCP / server logs).

**Tools exposed today:** see `mcp_server/server.py` (dataset meta, CSV features/classify/explain/screen, DREAMER analyze, `export_fhir_tool`, etc.).

---

## External A2A agents

- Po registers agents via URL ending in **`/.well-known/agent-card.json`** ([External Agents](https://docs.promptopinion.ai/agents/external-agents)).
- Follow **A2A v1** card shape ([A2A V1 Migration](https://docs.promptopinion.ai/a2a-v1-migration)).
- FHIR-capable agents declare the extension from [FHIR Context With A2A](https://docs.promptopinion.ai/fhir-context/a2a-fhir-context).
- **Consult** flow: Launchpad chat with a **BYO** agent → **Consult with another agent** → external agent ([External Agents](https://docs.promptopinion.ai/agents/external-agents)).

**STRAIN + A2A:** implement a small deployed agent (e.g. fork [**po-adk-python**](https://github.com/prompt-opinion/po-adk-python)) that calls FastAPI (`/api/...`) or imports `strain`, then register in Po.

---

## Suggested build order

1. Po account + model + patient ([Getting Started](https://docs.promptopinion.ai/getting-started)).  
2. STRAIN MCP **SSE + ngrok** → **Configuration → MCP Servers** → **BYO agent Tools** → Launchpad.  
3. STRAIN **FHIR bundle** improvements (`POST /api/export/fhir`, DREAMER parity) — see [hackathon-remaining-tasks.md](./hackathon-remaining-tasks.md).  
4. Optional: **`ai.promptopinion/fhir-context`** in MCP `initialize` + read Po headers in tools.  
5. Optional: **po-adk-python** external agent with v1 card.

## Community

- Discord / challenge links from [promptopinion.ai](https://www.promptopinion.ai/).  
- **po-overview** issues for cross-repo questions.
