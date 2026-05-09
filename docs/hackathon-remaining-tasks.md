# STRAIN hackathon plan — remaining tasks

This list tracks **STRAIN** vs the original product spec in [`.cursor/hackathon_idea_EMOTISCAN_v2.md`](../.cursor/hackathon_idea_EMOTISCAN_v2.md) (historical filename) and vs **Prompt Opinion “Agents Assemble”** expectations.

**Official Prompt Opinion docs (use these for registration & wiring):**

| Topic | Doc |
|-------|-----|
| Account, Gemini key, patient, agent, conversation | [Getting Started](https://docs.promptopinion.ai/getting-started) |
| LLM configs (`Configuration → Models`) | [Model Configuration](https://docs.promptopinion.ai/model-configuration) |
| A2A **v1** agent card (`supportedInterfaces`, removed `url`) | [A2A V1 Migration](https://docs.promptopinion.ai/a2a-v1-migration) |
| Workspace / Patient / Group scopes | [Agent Scopes](https://docs.promptopinion.ai/agents/agent-scopes) |
| **BYO agents** — system prompt, **Tools (MCP)**, A2A, FHIR extension | [BYO Agents](https://docs.promptopinion.ai/agents/byo-agents) |
| Register external A2A agents (`/.well-known/agent-card.json`) | [External Agents](https://docs.promptopinion.ai/agents/external-agents) |
| FHIR URL / token / patient id passed to tools | [FHIR Context Overview](https://docs.promptopinion.ai/fhir-context/overview) |
| **MCP + FHIR** — extension `ai.promptopinion/fhir-context`, headers | [FHIR Context With MCP](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context) |
| **A2A + FHIR** — extension URI + message metadata | [FHIR Context With A2A](https://docs.promptopinion.ai/fhir-context/a2a-fhir-context) |

Repo-specific guides: [prompt-opinion-hackathon.md](./prompt-opinion-hackathon.md) · [mcp-setup-first-steps.md](./mcp-setup-first-steps.md).

---

## Prompt Opinion — include STRAIN MCP in your hackathon app

“Your app” on Po means a **BYO agent** in the workspace that uses **this repo’s MCP server** under **Agents → BYO Agents → Tools** (per [BYO Agents](https://docs.promptopinion.ai/agents/byo-agents)).

### One-time Po setup

1. **LLM** — Create a key (e.g. Google AI Studio) and add a **model configuration** under **`Configuration → Models`** ([Model Configuration](https://docs.promptopinion.ai/model-configuration)).
2. **Account & workspace** — Register and complete [Getting Started](https://docs.promptopinion.ai/getting-started) (patient, enable an agent, conversation).
3. **Scopes** — Decide **Workspace** vs **Patient** vs **Group** for your demo ([Agent Scopes](https://docs.promptopinion.ai/agents/agent-scopes)).

### Register STRAIN as an MCP server (SSE + public URL)

1. On your machine: follow [mcp-setup-first-steps.md](./mcp-setup-first-steps.md) — `pip install -e .`, data + models, then run **SSE** (e.g. `./scripts/run_mcp_sse.sh` with `STRAIN_MCP_RELAX_DNS=1`).
2. Expose the port with **ngrok** (or similar): e.g. `ngrok http 8765`.
3. In Po: **`Configuration → MCP Servers`** → add server → enter base URL; Po sends **`initialize`** ([FHIR Context With MCP](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context)). Use the HTTPS URL + Po/FastMCP **SSE path** (often **`/sse`** — confirm in server logs / FastMCP docs).
4. If Po shows FHIR-context options, grant scopes you need after reviewing trust ([mcp-fhir-context](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context)).

### Attach MCP to a BYO agent

1. **`Agents → BYO Agents`** → **Add AI Agent** ([BYO Agents](https://docs.promptopinion.ai/agents/byo-agents)).
2. Pick your **model configuration**.
3. Open the **Tools** tab and **select the MCP server** you registered (STRAIN / `strain-tools`).
4. Optional: enable **A2A & Skills**, Po **FHIR context extension** for agents that should receive FHIR URL/token/patient from Po.
5. Use **Launchpad** to run conversations with that agent.

### External A2A agent (advanced)

- Deploy an agent that exposes **`/.well-known/agent-card.json`** per **A2A v1** ([External Agents](https://docs.promptopinion.ai/agents/external-agents), [A2A V1 Migration](https://docs.promptopinion.ai/a2a-v1-migration)).
- In Po: **`Agents → External Agents`** → **Add Connection** with the agent-card URL.
- **Consultation:** start from a **BYO** chat, then **Consult with another agent** → pick the external agent ([External Agents](https://docs.promptopinion.ai/agents/external-agents)).

### Code gaps for full Po FHIR ↔ MCP integration

- [ ] **`ai.promptopinion/fhir-context`** — Declare this extension in MCP **`initialize` → `capabilities.extensions`** and read **`X-FHIR-Server-URL`**, **`X-FHIR-Access-Token`**, **`X-Patient-ID`** (and optional refresh headers) in tool handlers when Po sends them ([mcp-fhir-context](https://docs.promptopinion.ai/fhir-context/mcp-fhir-context)). *Not implemented in `mcp_server/server.py` today.*
- [ ] **Tool logic** — Optionally POST screening bundles to the workspace FHIR server using those headers (demo only; align disclaimers).

---

## Recently implemented (repo)

| Area | What landed |
|------|----------------|
| **FHIR (demo)** | [`strain/io/fhir.py`](../strain/io/fhir.py) — `generate_fhir_bundle()`. |
| **MCP** | [`export_fhir_tool`](../mcp_server/server.py) — CSV → screening → FHIR JSON string. |
| **MCP transport** | Stdio + SSE CLI (`STRAIN_MCP_TRANSPORT`, etc.). |
| **3D UI** | [`Brain3D.tsx`](../backend/frontend/src/components/Brain3D.tsx). |
| **VAD / DREAMER pipeline** | Export script, Ridge VAD, `/api/analyze/dreamer`, per-electrode Welch in API. |
| **FastAPI `/api` prefix** | Matches Vite proxy paths. |

---

## Fix / complete next (high impact for judges)

1. **HTTP FHIR export** — `POST /api/export/fhir` (+ optional UI download).
2. **`export_fhir_tool` for DREAMER** — Same bundle shape from `analyze_dreamer_epoch`.
3. **FHIR validation** — `fhir.resources` / profiling cleanup.
4. **Po MCP FHIR extension** — Implement `ai.promptopinion/fhir-context` + header handling (see above).

---

## Hackathon submission checklist (non-code)

- [ ] Po **model** configured; **patient** created ([Getting Started](https://docs.promptopinion.ai/getting-started)).
- [ ] **BYO agent** with STRAIN **MCP** attached; tested on **Launchpad**.
- [ ] **Disclaimer** copy everywhere (not a medical device).
- [ ] **Marketplace / Studio** publish steps per challenge brief.
- [ ] Screen recording or live **Launchpad** demo path scripted (~5 min).

---

## Stretch — original v2 spec (still open)

### Data & personas

- [ ] DEAP / SEED loaders · `load_persona` MCP · EEG upload API/UI.

### Synthetic & generative

- [ ] WGAN-GP · `generate_synthetic` · `interpolate_emotional_trajectory` MCP tools.

### Models & screening depth

- [ ] Full asymmetry / multi-segment `screen_mental_health` · PyTorch classifiers beyond Ridge/logistic.

### Explainability

- [ ] LIME / temporal-frequency maps.

### MCP parity

- [ ] `generate_brain_map` JSON · `generate_frequency_visualization` · optional YAML tool catalog.

### A2A & multi-agent

- [ ] **`po-adk-python`** (or TS) deployment with **v1** agent card ([A2A V1 Migration](https://docs.promptopinion.ai/a2a-v1-migration)); FHIR extension `https://app.promptopinion.ai/schemas/a2a/v1/fhir-context` ([a2a-fhir-context](https://docs.promptopinion.ai/fhir-context/a2a-fhir-context)).
- [ ] Five logical agents vs one orchestrator — only if brief requires breadth.

### UI polish

- [ ] Tailwind/shadcn · persona picker · demo flow.

---

## Suggested order (next 2 weeks)

1. **Stable Po demo:** SSE MCP + ngrok → **Configuration → MCP Servers** → **BYO agent Tools** → Launchpad.  
2. **HTTP FHIR + DREAMER FHIR** in API/MCP.  
3. **Po FHIR MCP extension** in `mcp_server` (initialize + headers).  
4. **`fhir.resources` validation.**  
5. Optional **external A2A** agent via **`po-adk-python`** calling FastAPI or `strain`.

---

*Regenerate when major features land.*
