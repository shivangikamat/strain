# STRAIN Brain Monitor Redesign — Design Spec
**Date:** 2026-05-11  
**Audience:** Hackathon judges  
**Status:** Approved for implementation

---

## Overview

Transform STRAIN from a raw CSV/epoch prediction tool into a polished brain monitoring product demo. The app presents as a real-time EEG analysis device — hiding all dataset/CSV language — while using pre-selected DREAMER epochs under the hood.

---

## Visual Identity

- **Style:** Sci-fi neural interface (dark, glowing, dramatic)
- **Background:** `#13111C` with CSS grid overlay (existing)
- **Gradient pillar:** purple `#a855f7` → yellow `#fbbf24` (existing body::before, retained)
- **Accent colors:** `--accent: #a855f7` (purple), `--accent-secondary: #fbbf24` (yellow)
- **Fonts:** Outfit (headings/numbers), Inter (body) — existing
- **Panels:** `rgba(15,13,22,0.7)` with `backdrop-filter: blur(20px)`, `border-radius: 20px`
- **Top accent lines on cards:** 2px gradient top border (purple→yellow, cyan→purple, or yellow→orange per card type)

---

## Application Flow (3 Screens)

### Screen 1 — Connect Device (`ConnectScreen.tsx`)

**Layout:** Full-page centered column. No scrolling.

**Sections (top to bottom):**
1. **Nav bar** — STRAIN logo + animated status pill `"Searching for device..."` (yellow pulse dot)
2. **Hero text** — `"Connect your EEG headset"` + subtitle about EMOTIV EPOC+ placement
3. **Device animation** — 3 expanding concentric rings (CSS, purple, 2.5s stagger) around a glowing brain emoji icon box
4. **Scanning label** — `"● Scanning for device on port COM3..."` (blinking, purple)
5. **Patient profile cards** — horizontal row of 3 cards, each selectable (toggle `selected` class)
6. **CTA button** — `"Begin Neural Scan →"` gradient button (disabled until a patient is selected)
7. **Footer note** — `"Device detected · 14 channels active · 128 Hz sampling"` (appears after auto-detect delay)

**Patient profiles (3 pre-crafted):**

| Profile | Name | Tag | Description | DREAMER epoch |
|---------|------|-----|-------------|---------------|
| P1 | Alex Chen | High Stress | 32yo engineer, anxiety, poor sleep | To be picked: high arousal, mid-low valence epoch |
| P2 | Maria Santos | Calm & Focused | 28yo artist, meditative, alpha-dominant | To be picked: low arousal, high valence epoch |
| P3 | James O'Brien | Elevated Arousal | 58yo executive, high beta, cognitive load | To be picked: high arousal, high dominance epoch |

**Auto-detect simulation:** After 2.5s, status pill changes from `"Searching..."` to `"Device Found!"` (green). CTA becomes fully opaque.

**State:** Selected patient index stored in parent `App.tsx`. Clicking "Begin Neural Scan" transitions to Screen 2.

---

### Screen 2 — Scanning (`ScanScreen.tsx`)

**Layout:** Full-page two-column split. Left: brain + status. Right: channel feed panel.

**Left column:**
- Full-screen Brain3D component (Three.js, existing `Brain3D.tsx`) at center
- **Dendrite enhancement:** Each of the 14 electrode nodes emits 3–5 thin `THREE.Line` objects (tubular geometry, radius ~0.004, length 0.15–0.35, random directions within a hemisphere facing outward from the scalp). Lines pulse opacity in sync with the electrode's `useFrame` animation.
- Below brain: scan title `"Analyzing neural patterns"` + subtitle listing techniques (Welch PSD, Ridge regression, VAD estimation)
- Progress bar: `linear-gradient(90deg, #a855f7, #fbbf24)`, animates 0→100% over 9 seconds, then auto-advances to Screen 3

**Right column (340px, dark panel):**
- Header: `"14-Channel EEG Feed"` + `"● LIVE"` badge (blinking purple)
- Scrollable list of 14 channel rows: channel name | mini SVG wave | power value (color-coded: yellow=high >1.8, green=low <0.85, purple=mid)
- Bottom phase tracker: 7 steps (Device handshake → Signal quality → Epoch extraction → Welch PSD → VAD regression → Mental health screening → Report generation). First 3 done (green strikethrough), active step pulses purple, remaining are dimmed.

**Transition:** After progress bar completes (9s), automatically push to Screen 3. No user interaction required — judges watch it run.

---

### Screen 3 — Results (`ResultsScreen.tsx`)

**Layout:** Scrollable page. Nav + patient header + metric row + 3-col grid + 2-col bottom grid.

**Nav:** Logo + `"✓ Scan complete"` green badge + `"Download Report"` button + `"↺ New Scan"` button (returns to Screen 1).

**Patient header:** Avatar emoji + name + subtitle (age, profession, epoch/subject/trial) + scan timestamp.

**Top metric row (4 cards):**
1. Valence (yellow top border) — predicted value /5, badge
2. Arousal (purple top border) — predicted value /5, badge
3. Dominance (cyan top border) — predicted value /5, badge
4. Emotion State (green top border) — POSITIVE/NEUTRAL/NEGATIVE, classifier confidence %

**Main 3-col grid:**
- Col 1: Brain3D component (smaller, ~280px height) with live bandMeanPower from DREAMER epoch
- Col 2: VAD comparison bars (true vs predicted, cyan vs purple) + Affective Space SVG (mood meter, existing MoodMeter component)
- Col 3: 14-channel activity bar list + θ/α and β/α spectral ratio chips

**Bottom 2-col grid:**
- Col 1: Mental health screening — depression risk bar, anxiety risk bar, cognitive load bar (new, derived from β/α), recommendation box (yellow), non-clinical disclaimer
- Col 2: AI Explanation — natural language paragraph + top 5 contributing features with horizontal bars (yellow=positive, purple=negative contributions)

**Download Report:** calls `window.print()` (existing behavior).

---

## Backend Changes

### New endpoint: `GET /api/demo-patients`

Returns the 3 pre-crafted patient profiles with their metadata and pre-selected DREAMER epoch index.

```json
[
  {
    "id": "alex-chen",
    "name": "Alex Chen",
    "avatar": "👨‍💼",
    "age": 32,
    "profession": "Software Engineer",
    "tag": "High Stress",
    "description": "Reports elevated anxiety and poor sleep over the past 3 months.",
    "epoch_index": <chosen_int>,
    "accent": "purple-yellow"
  },
  ...
]
```

**Epoch selection:** Run `python scripts/pick_demo_epochs.py` (new script) to scan DREAMER epochs and find:
- P1 (Alex): epoch with valence 2.5–3.5, arousal >3.5 (stressed, activated)
- P2 (Maria): epoch with valence >4.0, arousal <2.5 (calm, positive)
- P3 (James): epoch with dominance >4.0, arousal >3.5 (in-control, activated)

Epoch indices are hardcoded in `api/main.py` as `DEMO_PATIENTS` list constant after running the picker script once.

### Cognitive load score

`strain/screening/mental_health.py` — add `cognitive_load` key to returned dict, derived from `beta_alpha` spectral ratio: `min(100, int(beta_alpha / 3.0 * 100))`.

---

## Frontend Architecture

```
src/
  App.tsx                  — phase state machine ('connect'|'scanning'|'results'), selectedPatient
  components/
    ConnectScreen.tsx      — Screen 1
    ScanScreen.tsx         — Screen 2 (uses Brain3D full-screen)
    ResultsScreen.tsx      — Screen 3
    Brain3D.tsx            — MODIFIED: add dendrite lines
    MoodMeter.tsx          — unchanged
  index.css                — unchanged
  App.css                  — add new component styles
```

**State in App.tsx:**
```ts
type Phase = 'connect' | 'scanning' | 'results'
type DemoPatient = { id, name, avatar, age, profession, tag, description, epoch_index, accent }

const [phase, setPhase] = useState<Phase>('connect')
const [patient, setPatient] = useState<DemoPatient | null>(null)
const [scanData, setScanData] = useState<DreamerAnalyzeResponse | null>(null)
```

Scan API call fires when `ScanScreen` mounts. Result stored in `scanData`, passed to `ResultsScreen` as prop.

---

## Spec Self-Review

- No TBDs or placeholders except epoch indices (resolved by picker script before implementation)
- Architecture consistent with Option C decision (no router, state in App.tsx)
- Three.js dendrite spec is concrete (Line objects, radius, length range, opacity pulse)
- Cognitive load formula defined
- Demo patients table has clear selection criteria for epoch picker script
- Scope is single frontend + minor backend addition — fits one implementation plan
