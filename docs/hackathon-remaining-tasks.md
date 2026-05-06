# STRAIN hackathon plan — remaining tasks

This list is **updated for the current codebase** (as of the latest changes). It compares the repo to the original v2 spec in [`.cursor/hackathon_idea_EMOTISCAN_v2.md`](../.cursor/hackathon_idea_EMOTISCAN_v2.md) (historical doc name).

---

## Recently implemented (adjust priorities accordingly)

| Area | What landed |
|------|----------------|
| **FHIR (demo)** | [`strain/io/fhir.py`](../strain/io/fhir.py) — `generate_fhir_bundle()` (Collection bundle: `DiagnosticReport`, `RiskAssessment` ×2, `Observation`). |
| **MCP** | [`export_fhir_tool`](../mcp_server/server.py) — CSV row → screening → FHIR JSON string. |
| **MCP transport** | CLI `--sse` / `--port` for local SSE in addition to env-based transport. |
| **3D UI** | [`Brain3D.tsx`](../backend/frontend/src/components/Brain3D.tsx) — R3F + Drei, DREAMER-style electrode layout, pulsing markers. |
| **Valence–arousal UI** | [`MoodMeter.tsx`](../backend/frontend/src/components/MoodMeter.tsx) — quadrant plot wired in [`App.tsx`](../backend/frontend/src/App.tsx). |
| **Frontend deps** | `three`, `@react-three/fiber`, `@react-three/drei` in [`package.json`](../backend/frontend/package.json). |
| **Brain3D API contract** | DREAMER `/analyze/dreamer` now includes per-electrode Welch means in `features.band_mean_power` (`beta_AF3`, …) via [`band_powers_welch`](../strain/features/eeg_epoch.py) + [`dreamer_analyze`](../strain/pipelines/dreamer_analyze.py) (`channel_names` from manifest). |

---

## Fix / complete next (small, high impact)

1. **FHIR outside MCP only** — Add **`POST /api/export/fhir`** (and optional query params for `row_index`, `patient_id`, `source`) so the web UI and Prompt Opinion can call HTTP without MCP. Optionally add a **Download JSON** button in `App.tsx`.
2. **`export_fhir_tool` for DREAMER** — When `source=dreamer`, bundle **VAD + screening** from `analyze_dreamer_epoch` using the same `generate_fhir_bundle` shape (or a parallel builder for continuous scores).
3. **FHIR validation** — Use **`fhir.resources`** (and optionally HAPI) to validate bundles before return; tighten R4 fields (`RiskAssessment`, `Observation.value*`) per profiling docs.

---

## Data & personas (plan §2–3, Phase 1)

- [ ] **DEAP / SEED loaders** — `load_dataset` returning `(trials, channels, time)`, `sfreq`, `channel_names` (not only Kaggle CSV + DREAMER export).
- [ ] **`load_persona`** MCP + data — Alex / Jordan / … curated segments tied to real `.mat` trials once DEAP/SEED are available.
- [ ] **EEG upload path** — Accept user `.edf` / `.csv` in API + UI (plan demo flow).

---

## Synthetic & generative (plan §2.2, Phase 2)

- [ ] **WGAN-GP** training + checkpoint.
- [ ] **`generate_synthetic`** MCP tool.
- [ ] **`interpolate_emotional_trajectory`** MCP tool + optional UI animation.

---

## Features & models (Phases 3–4)

- [ ] **Per-channel features** — DE, PSD, HOC, **asymmetry** on multi-channel epochs (DREAMER/DEAP), not only global Welch + CSV proxies.
- [ ] **Classifier options** — `bi_hemispheric` / `4d_crnn` / TSS-style model (PyTorch), not only logistic + Ridge VAD.
- [ ] **`screen_mental_health` spec** — Multi-segment `session_data`, longitudinal mode, full biomarker objects (FAA, θ/β, flags) per plan YAML.

---

## Explainability (Phase 5)

- [ ] **LIME** (or Captum) for the deployed classifier, with **temporal** and **frequency** importance maps aligned to plan `explain_decision` outputs.

---

## MCP parity (plan §3.2, Phase 6)

- [ ] **`load_persona`**, **`generate_synthetic`**, **`interpolate_emotional_trajectory`**.
- [ ] **`generate_brain_map`** — Return JSON mesh / electrode activations for non-React clients (UI already has a 3D view).
- [ ] **`generate_frequency_visualization`** — SVG / structured band-by-region output.
- [ ] **Optional:** machine-readable **YAML tool catalog** mirroring the plan (for judges / SHARP docs).

---

## A2A & platform (plan §3.3, Phase 7 + Prompt Opinion)

- [ ] **Fork / adapt [`po-adk-python`](https://github.com/prompt-opinion/po-adk-python)** — One A2A v1 agent exposing STRAIN analysis + FHIR export; register with Prompt Opinion (`agent-card`, `X-API-Key`, `PO_PLATFORM_BASE_URL`).
- [ ] **Five logical agents** — Either separate deployments or one orchestrator with distinct **agent cards** / skills (Data Curator, Analysis, Explainer, Synthetic, Visualization / Care Navigator).

---

## 3D & viz polish (plan §4, Phase 8)

- [ ] **True topography** — Surface mesh or better head model; channel values from real PSD pipeline.
- [ ] **Hemispheric split view**, **frequency band panel**, **emotion trajectory** animation (plan storyboard).

---

## UI / demo (plan §6, Phase 10)

- [ ] **Tailwind + shadcn** (plan stack) — optional polish pass.
- [ ] **Persona picker** + scripted **5-minute demo** flow in the app.
- [ ] **Marketplace / Launchpad** rehearsal — ops checklist (not code).

---

## Suggested order (next sprints)

1. **HTTP FHIR export** + **DREAMER FHIR** in MCP.  
2. **fhir.resources** validation + tighten resource shapes.  
3. **DEAP loader** + one **persona** + **`load_persona`**.  
4. **LIME** + **`generate_brain_map`** JSON for MCP.  
5. **po-adk-python** external agent.

---

*Regenerate this file when major features land so the team has a single checklist.*
