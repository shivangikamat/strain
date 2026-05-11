# Brain Monitor Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform STRAIN's frontend into a polished 3-screen brain monitor demo (Connect → Scan → Results) using pre-selected DREAMER epochs, hiding all CSV/dataset language.

**Architecture:** Phase state machine in `App.tsx` renders one of three full-page screen components. Backend gains a `/api/demo-patients` endpoint returning 3 hardcoded patient profiles (with DREAMER epoch indices). Brain3D gains dendrite lines per electrode. All data types live in a shared `types.ts`.

**Tech Stack:** React 19, TypeScript, Three.js / @react-three/fiber, Recharts, FastAPI, existing sklearn Ridge VAD model.

---

## File Map

### New files
- `scripts/pick_demo_epochs.py` — one-time script to find ideal epoch indices for each patient profile
- `backend/frontend/src/types.ts` — all shared TypeScript types (DemoPatient, DreamerAnalyzeResponse, Phase)
- `backend/frontend/src/components/ConnectScreen.tsx` — Screen 1: device connect + patient select
- `backend/frontend/src/components/ScanScreen.tsx` — Screen 2: 3D brain scan animation
- `backend/frontend/src/components/ResultsScreen.tsx` — Screen 3: full results dashboard

### Modified files
- `strain/screening/mental_health.py` — add `cognitive_load` key to return dict
- `strain/models/dreamer_vad.py` — add `beta_alpha` param + `cognitive_load` to `dreamer_vad_screening`
- `strain/pipelines/dreamer_analyze.py` — pass `beta_alpha` from features into `dreamer_vad_screening`
- `api/main.py` — add `DEMO_PATIENTS` constant + `GET /api/demo-patients` route
- `backend/frontend/src/components/Brain3D.tsx` — add `height` prop + `DendriteLines` per electrode
- `backend/frontend/src/App.tsx` — replace with phase state machine (thin shell, ~30 lines)
- `backend/frontend/src/App.css` — append styles for all new screen components
- `tests/test_health.py` — add test for new `/api/demo-patients` endpoint

---

## Task 1: Epoch Picker Script

**Files:**
- Create: `scripts/pick_demo_epochs.py`

This is a one-time run script. It scans the exported DREAMER epochs, predicts VAD for a sample, and prints the best epoch index for each patient profile. Run it, copy the indices, use them in Task 3.

- [ ] **Step 1: Create the picker script**

```python
# scripts/pick_demo_epochs.py
"""
Find ideal DREAMER epoch indices for the 3 demo patient profiles.

Run after exporting DREAMER epochs and training the VAD model:
  python scripts/pick_demo_epochs.py

Prints recommended epoch indices. Copy them into DEMO_PATIENTS in api/main.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from strain.data.dreamer_epochs import (
    dreamer_processed_dir,
    load_dreamer_manifest,
    open_dreamer_X_memmap,
)
from strain.features.dreamer_featurize import featurize_dreamer_epoch
from strain.models.dreamer_vad import load_dreamer_vad_bundle

SAMPLE_EVERY = 5  # check every Nth epoch for speed


def _score_profile(v: float, a: float, d: float, profile: str) -> float:
    """Higher = better match for the profile."""
    if profile == "stressed":
        # High arousal, mid-low valence
        return a - abs(v - 2.8)
    if profile == "calm":
        # High valence, low arousal
        return v - a
    if profile == "executive":
        # High dominance, high arousal
        return d + a * 0.5
    return 0.0


def main() -> None:
    base = dreamer_processed_dir(None)
    meta = load_dreamer_manifest(base)
    sfreq = float(meta["sfreq"])
    X = open_dreamer_X_memmap(base, mode="r")
    n = int(X.shape[0])

    try:
        bundle = load_dreamer_vad_bundle()
    except FileNotFoundError:
        print("ERROR: VAD model not found. Run: python scripts/train_dreamer_vad.py")
        sys.exit(1)

    pipeline = bundle["pipeline"]
    v_all = np.load(str(base / "valence.npy"), mmap_mode="r")
    a_all = np.load(str(base / "arousal.npy"), mmap_mode="r")
    d_all = np.load(str(base / "dominance.npy"), mmap_mode="r")

    profiles = {"stressed": (-1, -999.0), "calm": (-1, -999.0), "executive": (-1, -999.0)}

    indices = range(0, n, SAMPLE_EVERY)
    print(f"Scanning {len(list(indices))} epochs (every {SAMPLE_EVERY} of {n})...")

    for i in indices:
        eeg = np.asarray(X[i], dtype=np.float64)
        feats = featurize_dreamer_epoch(eeg, sfreq).reshape(1, -1)
        pred = pipeline.predict(feats)[0]
        pv, pa, pd = float(pred[0]), float(pred[1]), float(pred[2])

        for name in profiles:
            score = _score_profile(pv, pa, pd, name)
            if score > profiles[name][1]:
                profiles[name] = (i, score)

    print("\n=== Recommended epoch indices ===")
    print(f"Alex Chen    (stressed):  epoch {profiles['stressed'][0]}")
    print(f"  true VAD: V={v_all[profiles['stressed'][0]]:.2f} "
          f"A={a_all[profiles['stressed'][0]]:.2f} "
          f"D={d_all[profiles['stressed'][0]]:.2f}")
    print()
    print(f"Maria Santos (calm):      epoch {profiles['calm'][0]}")
    print(f"  true VAD: V={v_all[profiles['calm'][0]]:.2f} "
          f"A={a_all[profiles['calm'][0]]:.2f} "
          f"D={d_all[profiles['calm'][0]]:.2f}")
    print()
    print(f"James O'Brien (executive): epoch {profiles['executive'][0]}")
    print(f"  true VAD: V={v_all[profiles['executive'][0]]:.2f} "
          f"A={a_all[profiles['executive'][0]]:.2f} "
          f"D={d_all[profiles['executive'][0]]:.2f}")
    print()
    print("Copy these indices into DEMO_PATIENTS in api/main.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the picker (requires exported DREAMER + trained VAD model)**

```bash
source .venv/bin/activate
python scripts/pick_demo_epochs.py
```

Expected output (values will differ):
```
Scanning 800 epochs (every 5 of 4000)...

=== Recommended epoch indices ===
Alex Chen    (stressed):  epoch 1235
  true VAD: V=2.80 A=4.20 D=2.50

Maria Santos (calm):      epoch 872
  true VAD: V=4.50 A=1.80 D=3.20

James O'Brien (executive): epoch 310
  true VAD: V=3.80 A=4.10 D=4.40

Copy these indices into DEMO_PATIENTS in api/main.py
```

Note the 3 epoch indices. Use them in Task 3. If DREAMER is not available, use `0`, `100`, `200` as fallbacks — the demo will still run with different data per profile.

- [ ] **Step 3: Commit the picker script**

```bash
git add scripts/pick_demo_epochs.py
git commit -m "feat: add demo epoch picker script"
```

---

## Task 2: Add cognitive_load to Screening

**Files:**
- Modify: `strain/screening/mental_health.py`
- Modify: `strain/models/dreamer_vad.py` (function `dreamer_vad_screening`)
- Modify: `strain/pipelines/dreamer_analyze.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_health.py`:

```python
def test_screen_mental_health_has_cognitive_load() -> None:
    from strain.screening.mental_health import screen_mental_health

    cls = {"probabilities": {"NEGATIVE": 0.1, "NEUTRAL": 0.2, "POSITIVE": 0.7}, "confidence": 0.7}
    feats = {"spectral_ratios": {"beta_alpha": 2.1, "theta_alpha": 0.9}}
    result = screen_mental_health(cls, feats)
    assert "cognitive_load" in result
    assert 0 <= result["cognitive_load"]["score"] <= 100


def test_dreamer_vad_screening_has_cognitive_load() -> None:
    from strain.models.dreamer_vad import dreamer_vad_screening

    pred = {"valence": 3.0, "arousal": 4.0, "dominance": 2.5}
    result = dreamer_vad_screening(pred, beta_alpha=2.1)
    assert "cognitive_load" in result
    assert 0 <= result["cognitive_load"]["score"] <= 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_health.py::test_screen_mental_health_has_cognitive_load tests/test_health.py::test_dreamer_vad_screening_has_cognitive_load -v
```

Expected: FAIL — `KeyError: 'cognitive_load'`

- [ ] **Step 3: Update `strain/screening/mental_health.py`**

Add `cognitive_load` to the returned dict. The score maps `beta_alpha` to 0–100 (`beta_alpha=3.0` → 100%).

```python
def screen_mental_health(
    classification: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probs = classification.get("probabilities", {})
    p_neg = float(probs.get("NEGATIVE", 0.0))
    p_neu = float(probs.get("NEUTRAL", 0.0))
    p_pos = float(probs.get("POSITIVE", 0.0))

    ratios = (features or {}).get("spectral_ratios", {})
    beta_alpha = float(ratios.get("beta_alpha", 1.0))

    depression_risk = min(100.0, max(0.0, 55.0 * p_neg + 8.0 * max(0.0, beta_alpha - 1.0)))
    anxiety_risk = min(100.0, max(0.0, 45.0 * (1.0 - p_neu) + 10.0 * max(0.0, beta_alpha - 1.0)))
    cognitive_load = min(100.0, max(0.0, beta_alpha / 3.0 * 100.0))

    rec = "no_concern"
    if depression_risk > 70 or anxiety_risk > 70:
        rec = "seek_specialist"
    elif depression_risk > 50 or anxiety_risk > 50:
        rec = "consult_pcp"
    elif depression_risk > 35 or anxiety_risk > 35:
        rec = "monitor"

    return {
        "disclaimer": (
            "Demonstration scores only — not a medical device. "
            "Do not use for diagnosis or treatment decisions."
        ),
        "depression_risk": {"score": depression_risk, "confidence": classification.get("confidence")},
        "anxiety_risk": {"score": anxiety_risk, "confidence": classification.get("confidence")},
        "cognitive_load": {"score": cognitive_load},
        "recommendation": rec,
        "key_findings": [
            f"Emotion model: NEGATIVE {p_neg:.2f}, NEUTRAL {p_neu:.2f}, POSITIVE {p_pos:.2f}.",
            f"Proxy beta/alpha ratio: {beta_alpha:.3f}.",
        ],
    }
```

- [ ] **Step 4: Update `dreamer_vad_screening` in `strain/models/dreamer_vad.py`**

Replace the existing `dreamer_vad_screening` function (lines 198–217) with:

```python
def dreamer_vad_screening(
    pred: dict[str, float],
    beta_alpha: float | None = None,
) -> dict[str, Any]:
    """Map continuous VAD (+ optional beta_alpha) to demo risk scores (non-clinical)."""
    v = pred["valence"]
    a = pred["arousal"]
    d = pred["dominance"]
    dep = max(0.0, min(100.0, (3.0 - v) * 25.0 + max(0.0, a - 3.0) * 10.0))
    anx = max(0.0, min(100.0, (a - 2.5) * 22.0 + abs(v - 3.0) * 5.0))
    if beta_alpha is not None:
        cog = min(100.0, max(0.0, beta_alpha / 3.0 * 100.0))
    else:
        cog = min(100.0, max(0.0, (a - 1.0) / 4.0 * 60.0 + (5.0 - d) / 4.0 * 40.0))
    rec = "no_concern"
    if dep > 65 or anx > 65:
        rec = "monitor"
    if dep > 80 or anx > 80:
        rec = "consult_pcp"
    return {
        "disclaimer": "Demonstration mapping from model VAD only — not clinical.",
        "depression_risk": {"score": dep, "note": "Higher when predicted valence is low."},
        "anxiety_risk": {"score": anx, "note": "Higher when predicted arousal is high."},
        "cognitive_load": {"score": cog},
        "recommendation": rec,
        "key_findings": [
            f"Predicted valence/arousal/dominance: {v:.2f} / {a:.2f} / {d:.2f}",
        ],
    }
```

- [ ] **Step 5: Pass beta_alpha into dreamer_vad_screening in `strain/pipelines/dreamer_analyze.py`**

Change the `screening = dreamer_vad_screening(pred_vad)` call (line 63) to:

```python
screening = dreamer_vad_screening(
    pred_vad,
    beta_alpha=feats["spectral_ratios"].get("beta_alpha"),
)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_health.py::test_screen_mental_health_has_cognitive_load tests/test_health.py::test_dreamer_vad_screening_has_cognitive_load -v
```

Expected: PASS

- [ ] **Step 7: Run full test suite to check no regressions**

```bash
pytest -q
```

Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add strain/screening/mental_health.py strain/models/dreamer_vad.py strain/pipelines/dreamer_analyze.py tests/test_health.py
git commit -m "feat: add cognitive_load to screening output"
```

---

## Task 3: Add /api/demo-patients Endpoint

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_health.py`:

```python
def test_demo_patients_returns_three_profiles() -> None:
    client = TestClient(app)
    r = client.get("/api/demo-patients")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    for p in data:
        assert "id" in p
        assert "name" in p
        assert "epoch_index" in p
        assert isinstance(p["epoch_index"], int)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_health.py::test_demo_patients_returns_three_profiles -v
```

Expected: FAIL — 404

- [ ] **Step 3: Add DEMO_PATIENTS constant and route to `api/main.py`**

After the imports block (after line 24, before `app = FastAPI(...)`), add:

```python
# Epoch indices chosen by scripts/pick_demo_epochs.py — update after running that script.
# Fallback: 0, 100, 200 if DREAMER not exported (demo still works, same data per run).
DEMO_PATIENTS: list[dict] = [
    {
        "id": "alex-chen",
        "name": "Alex Chen",
        "avatar": "👨‍💼",
        "age": 32,
        "profession": "Software Engineer",
        "tag": "High Stress",
        "description": "Reports elevated anxiety and poor sleep over the past 3 months.",
        "epoch_index": 0,   # replace with output from pick_demo_epochs.py
        "accent": "purple-yellow",
    },
    {
        "id": "maria-santos",
        "name": "Maria Santos",
        "avatar": "👩‍🎨",
        "age": 28,
        "profession": "Artist",
        "tag": "Calm & Focused",
        "description": "Meditative baseline. Strong alpha dominance and low arousal state.",
        "epoch_index": 100,  # replace with output from pick_demo_epochs.py
        "accent": "cyan-purple",
    },
    {
        "id": "james-obrien",
        "name": "James O'Brien",
        "avatar": "👴",
        "age": 58,
        "profession": "Executive",
        "tag": "Elevated Arousal",
        "description": "High dominance, elevated beta activity. Active cognitive load detected.",
        "epoch_index": 200,  # replace with output from pick_demo_epochs.py
        "accent": "yellow-orange",
    },
]
```

Then add the route inside the `api` router (after the existing routes, before `app.include_router(api)`):

```python
@api.get("/demo-patients")
def demo_patients() -> list[dict]:
    """Return the 3 pre-crafted demo patient profiles."""
    return DEMO_PATIENTS
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_health.py::test_demo_patients_returns_three_profiles -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest -q
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_health.py
git commit -m "feat: add /api/demo-patients endpoint with 3 patient profiles"
```

---

## Task 4: Shared TypeScript Types

**Files:**
- Create: `backend/frontend/src/types.ts`

- [ ] **Step 1: Create `src/types.ts`**

```typescript
// src/types.ts

export type Phase = 'connect' | 'scanning' | 'results'

export type DemoPatient = {
  id: string
  name: string
  avatar: string
  age: number
  profession: string
  tag: string
  description: string
  epoch_index: number
  accent: 'purple-yellow' | 'cyan-purple' | 'yellow-orange'
}

export type ScreeningResult = {
  disclaimer: string
  depression_risk: { score: number; note?: string }
  anxiety_risk: { score: number; note?: string }
  cognitive_load: { score: number }
  recommendation: string
  key_findings: string[]
}

export type DreamerAnalyzeResponse = {
  epoch_index: number
  subject_id: number
  trial_id: number
  true_vad: { valence: number; arousal: number; dominance: number }
  features: {
    spectral_ratios: { theta_alpha: number; beta_alpha: number }
    band_mean_power: Record<string, number>
  }
  predicted_vad: { valence: number; arousal: number; dominance: number } | null
  explanation: {
    natural_language_explanation: string
    per_target_top_features: Record<string, { name: string; contribution: number }[]>
  }
  screening: ScreeningResult | null
}
```

- [ ] **Step 2: Commit**

```bash
git add backend/frontend/src/types.ts
git commit -m "feat: add shared TypeScript types"
```

---

## Task 5: Brain3D Dendrite Enhancement

**Files:**
- Modify: `backend/frontend/src/components/Brain3D.tsx`

Add a `DendriteLines` component that renders 4 thin pulsing lines per electrode, and add a `height` prop to the container.

- [ ] **Step 1: Add `DendriteLines` component**

Insert this function **before** `ElectrodeMarker` in `Brain3D.tsx` (after line 41):

```tsx
function DendriteLines({
  origin,
  value,
  color,
}: {
  origin: [number, number, number]
  value: number
  color: string
}) {
  const matRef = useRef<THREE.LineBasicMaterial>(null)

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions: number[] = []
    const [ox, oy, oz] = origin
    const len = Math.sqrt(ox * ox + oy * oy + oz * oz) || 1
    const nx = ox / len
    const ny = oy / len
    const nz = oz / len

    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2
      const spread = 0.45
      const dx = nx + Math.cos(angle) * spread
      const dy = ny + Math.sin(angle) * spread
      const dz = nz + 0.05
      const dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1
      const length = 0.18 + (i % 3) * 0.08
      positions.push(
        ox, oy, oz,
        ox + (dx / dl) * length,
        oy + (dy / dl) * length,
        oz + (dz / dl) * length,
      )
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    return geo
  }, [origin])

  useFrame(({ clock }) => {
    if (matRef.current) {
      matRef.current.opacity =
        0.15 + Math.abs(Math.sin(clock.elapsedTime * value * 2.5 + origin[0])) * 0.45
    }
  })

  return (
    <lineSegments geometry={geometry} renderOrder={1}>
      <lineBasicMaterial ref={matRef} color={color} transparent opacity={0.3} />
    </lineSegments>
  )
}
```

- [ ] **Step 2: Update `ElectrodeMarker` to render dendrites**

Replace the existing `ElectrodeMarker` function (lines 42–71) with:

```tsx
function ElectrodeMarker({
  position,
  value,
}: {
  position: [number, number, number]
  value: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)

  const color = useMemo(() => {
    if (value > 2.0) return '#fbbf24'
    if (value > 1.2) return '#f97316'
    if (value > 0.6) return '#a855f7'
    return '#22c55e'
  }, [value])

  useFrame(({ clock }) => {
    if (meshRef.current) {
      const scale = 1.0 + Math.sin(clock.elapsedTime * value * 3) * 0.1
      meshRef.current.scale.set(scale, scale, scale)
    }
  })

  return (
    <group>
      <mesh position={position} ref={meshRef} renderOrder={2}>
        <sphereGeometry args={[0.048, 18, 18]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={Math.min(value * 0.45, 1.2)}
          toneMapped={false}
          depthTest
          depthWrite
        />
      </mesh>
      <DendriteLines origin={position} value={value} color={color} />
    </group>
  )
}
```

- [ ] **Step 3: Add `height` prop to the exported `Brain3D` component**

Replace the `Brain3D` export (lines 200–226) with:

```tsx
export function Brain3D({
  bandMeanPower,
  height = '280px',
}: {
  bandMeanPower: Record<string, number> | undefined
  height?: string | number
}) {
  return (
    <div
      style={{
        width: '100%',
        maxWidth: '100%',
        minWidth: 0,
        height,
        boxSizing: 'border-box',
        background: 'rgba(0,0,0,0.2)',
        borderRadius: '16px',
        border: '1px solid rgba(255,255,255,0.08)',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <Canvas
        camera={{ position: [0, 0.15, 3.25], fov: 42 }}
        gl={{ alpha: true, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true }}
        style={{ display: 'block', width: '100%', height: '100%', maxWidth: '100%' }}
        dpr={[1, 2]}
      >
        <SceneContent bandMeanPower={bandMeanPower} />
      </Canvas>
    </div>
  )
}
```

- [ ] **Step 4: Verify frontend builds without TypeScript errors**

```bash
cd backend/frontend && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add backend/frontend/src/components/Brain3D.tsx
git commit -m "feat: add dendrite lines to Brain3D electrodes, add height prop"
```

---

## Task 6: CSS for New Screens

**Files:**
- Modify: `backend/frontend/src/App.css`

- [ ] **Step 1: Append new styles to end of `App.css`**

```css
/* ─── ConnectScreen ─────────────────────────────────────────────── */

.connect-screen {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.connect-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem 3rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  position: relative;
  z-index: 10;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.9rem;
  border-radius: 100px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  transition: all 0.4s ease;
}

.status-pill.searching {
  background: rgba(251, 191, 36, 0.1);
  border: 1px solid rgba(251, 191, 36, 0.3);
  color: #fbbf24;
}

.status-pill.found {
  background: rgba(52, 211, 153, 0.1);
  border: 1px solid rgba(52, 211, 153, 0.3);
  color: #34d399;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  animation: status-pulse 1.5s ease-in-out infinite;
}

.status-pill.searching .status-dot { background: #fbbf24; }
.status-pill.found .status-dot { background: #34d399; }

@keyframes status-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.7); }
}

.connect-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem 1.5rem;
  gap: 2.5rem;
  position: relative;
  z-index: 1;
}

.connect-hero-text {
  text-align: center;
}

.connect-hero-text h1 {
  font-size: clamp(2rem, 4vw, 3rem);
  letter-spacing: -0.03em;
  margin-bottom: 0.5rem;
}

.connect-hero-text h1 span { color: var(--accent); }

.connect-hero-text p {
  font-size: 0.9rem;
  max-width: 440px;
  line-height: 1.6;
  margin: 0 auto;
}

.device-zone {
  position: relative;
  width: 160px;
  height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.device-rings {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.device-ring {
  position: absolute;
  border-radius: 50%;
  border: 1px solid rgba(168, 85, 247, 0.3);
  animation: ring-expand 2.5s ease-out infinite;
}

.device-ring:nth-child(1) { width: 72px; height: 72px; animation-delay: 0s; }
.device-ring:nth-child(2) { width: 112px; height: 112px; animation-delay: 0.6s; }
.device-ring:nth-child(3) { width: 150px; height: 150px; animation-delay: 1.2s; }

@keyframes ring-expand {
  0% { opacity: 0.8; transform: scale(0.8); }
  100% { opacity: 0; transform: scale(1.1); }
}

.device-icon-box {
  position: relative;
  z-index: 1;
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(251, 191, 36, 0.15));
  border: 1px solid rgba(168, 85, 247, 0.4);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  box-shadow: 0 0 30px rgba(168, 85, 247, 0.3);
}

.scan-label {
  font-size: 0.72rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  animation: blink-label 2s ease-in-out infinite;
}

@keyframes blink-label {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.patient-section {
  width: 100%;
  max-width: 800px;
}

.patient-section-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #52525b;
  text-align: center;
  margin-bottom: 1rem;
}

.patient-cards-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  justify-content: center;
}

.patient-card {
  background: rgba(15, 13, 22, 0.8);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 20px;
  padding: 1.25rem 1.5rem;
  width: 220px;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.patient-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  border-radius: 20px 20px 0 0;
}

.patient-card[data-accent="purple-yellow"]::before { background: linear-gradient(90deg, #a855f7, #fbbf24); }
.patient-card[data-accent="cyan-purple"]::before { background: linear-gradient(90deg, #06b6d4, #a855f7); }
.patient-card[data-accent="yellow-orange"]::before { background: linear-gradient(90deg, #fbbf24, #f97316); }

.patient-card:hover {
  border-color: rgba(168, 85, 247, 0.25);
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
}

.patient-card.selected {
  border-color: rgba(168, 85, 247, 0.55);
  background: rgba(168, 85, 247, 0.08);
  box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2), 0 20px 40px rgba(0, 0, 0, 0.5);
  transform: translateY(-4px);
}

.patient-avatar-box {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
  margin-bottom: 0.75rem;
  background: rgba(168, 85, 247, 0.12);
}

.patient-card-name {
  color: #fff;
  font-weight: 700;
  font-size: 1rem;
  margin-bottom: 0.2rem;
  font-family: var(--heading);
}

.patient-card-tag {
  display: inline-block;
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.15rem 0.5rem;
  border-radius: 100px;
  margin-bottom: 0.6rem;
  background: rgba(168, 85, 247, 0.12);
  color: #c084fc;
}

.patient-card-desc {
  font-size: 0.75rem;
  color: #71717a;
  line-height: 1.5;
}

.patient-card-meta {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 0.75rem;
}

.patient-chip {
  font-size: 0.58rem;
  padding: 0.2rem 0.5rem;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 6px;
  color: #52525b;
}

.connect-cta {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
}

.cta-primary-btn {
  background: linear-gradient(135deg, #a855f7, #fbbf24);
  border: none;
  color: #fff;
  font-weight: 700;
  padding: 0.9rem 3rem;
  border-radius: 100px;
  font-size: 0.95rem;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  box-shadow: 0 0 30px rgba(168, 85, 247, 0.4);
  transition: all 0.2s ease;
  font-family: var(--heading);
  opacity: 0.4;
  pointer-events: none;
}

.cta-primary-btn.ready {
  opacity: 1;
  pointer-events: auto;
}

.cta-primary-btn.ready:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 50px rgba(168, 85, 247, 0.6);
}

.connect-footnote {
  font-size: 0.68rem;
  color: #3f3f46;
}

/* ─── ScanScreen ─────────────────────────────────────────────────── */

.scan-screen {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.scan-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem 3rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  flex-shrink: 0;
}

.scan-patient-pill {
  background: rgba(168, 85, 247, 0.1);
  border: 1px solid rgba(168, 85, 247, 0.25);
  border-radius: 100px;
  padding: 0.35rem 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
}

.scan-patient-pill .spp-label { color: #71717a; }
.scan-patient-pill .spp-name { color: #fff; font-weight: 600; }

.scan-body {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 340px;
  min-height: 0;
}

.scan-brain-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  gap: 2rem;
}

.scan-status-block {
  text-align: center;
}

.scan-status-title {
  font-family: var(--heading);
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 0.35rem;
}

.scan-status-sub {
  font-size: 0.78rem;
  color: #71717a;
  animation: blink-label 2s ease-in-out infinite;
}

.scan-progress-wrap {
  width: 300px;
  margin-top: 0.5rem;
}

.scan-progress-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.65rem;
  color: #52525b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.4rem;
}

.scan-progress-pct { color: var(--accent); }

.scan-progress-track {
  height: 4px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 100px;
  overflow: hidden;
}

.scan-progress-fill {
  height: 100%;
  border-radius: 100px;
  background: linear-gradient(90deg, #a855f7, #fbbf24);
  width: 0%;
  transition: width 0.5s ease;
}

.scan-channel-col {
  border-left: 1px solid rgba(255, 255, 255, 0.05);
  background: rgba(10, 8, 15, 0.6);
  backdrop-filter: blur(20px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.scan-channel-header {
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #52525b;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.live-badge {
  background: rgba(168, 85, 247, 0.1);
  border: 1px solid rgba(168, 85, 247, 0.3);
  color: var(--accent);
  font-size: 0.55rem;
  padding: 0.15rem 0.5rem;
  border-radius: 100px;
  animation: blink-label 1.5s ease-in-out infinite;
}

.scan-channel-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.4rem 0;
}

.channel-row {
  display: grid;
  grid-template-columns: 38px 1fr 38px;
  align-items: center;
  gap: 0.6rem;
  padding: 0.45rem 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.channel-name {
  font-size: 0.62rem;
  font-weight: 600;
  color: #71717a;
  font-family: var(--mono);
}

.channel-wave {
  height: 22px;
  overflow: hidden;
}

.channel-val {
  font-size: 0.58rem;
  font-family: var(--mono);
  text-align: right;
}

.channel-val.high { color: #fbbf24; }
.channel-val.mid  { color: #a855f7; }
.channel-val.low  { color: #34d399; }

.scan-phase-list {
  padding: 1rem 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  flex-shrink: 0;
}

.phase-item {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.38rem 0;
  font-size: 0.7rem;
  transition: all 0.3s ease;
}

.phase-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  transition: background 0.3s ease;
}

.phase-item.done .phase-dot   { background: #34d399; }
.phase-item.active .phase-dot { background: var(--accent); animation: status-pulse 1s ease-in-out infinite; }
.phase-item.waiting .phase-dot { background: rgba(255, 255, 255, 0.1); }

.phase-item.done .phase-label   { color: #52525b; text-decoration: line-through; }
.phase-item.active .phase-label { color: #fff; }
.phase-item.waiting .phase-label { color: #3f3f46; }

/* ─── ResultsScreen ──────────────────────────────────────────────── */

.results-screen {
  min-height: 100vh;
}

.results-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem 3rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(19, 17, 28, 0.9);
  backdrop-filter: blur(20px);
}

.results-nav-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.result-badge {
  background: rgba(52, 211, 153, 0.1);
  border: 1px solid rgba(52, 211, 153, 0.3);
  color: #34d399;
  font-size: 0.65rem;
  padding: 0.3rem 0.8rem;
  border-radius: 100px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.result-action-btn {
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #fff;
  font-size: 0.7rem;
  padding: 0.4rem 1rem;
  border-radius: 100px;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  transition: all 0.2s ease;
}

.result-action-btn:hover { background: rgba(255, 255, 255, 0.12); }

.rescan-btn {
  background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(251, 191, 36, 0.1));
  border: 1px solid rgba(168, 85, 247, 0.3);
  color: #c084fc;
  font-size: 0.7rem;
  padding: 0.4rem 1rem;
  border-radius: 100px;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  transition: all 0.2s ease;
}

.rescan-btn:hover { border-color: rgba(168, 85, 247, 0.5); }

.results-content {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem 2rem 4rem;
  position: relative;
  z-index: 1;
}

.patient-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
}

.patient-header-left {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.patient-header-avatar {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(251, 191, 36, 0.2));
  border: 1px solid rgba(168, 85, 247, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.6rem;
}

.patient-header-name {
  font-family: var(--heading);
  font-weight: 700;
  font-size: 1.4rem;
  color: #fff;
  margin-bottom: 0.15rem;
}

.patient-header-sub { font-size: 0.75rem; color: #71717a; }
.patient-header-time { font-size: 0.7rem; color: #52525b; }

.metric-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

@media (max-width: 900px) {
  .metric-row { grid-template-columns: repeat(2, 1fr); }
}

.metric-card {
  background: rgba(15, 13, 22, 0.7);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 16px;
  padding: 1.1rem 1.25rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s ease;
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}

.metric-card[data-color="yellow"]::before { background: linear-gradient(90deg, #fbbf24, #f97316); }
.metric-card[data-color="purple"]::before { background: linear-gradient(90deg, #a855f7, #c084fc); }
.metric-card[data-color="cyan"]::before   { background: linear-gradient(90deg, #06b6d4, #a855f7); }
.metric-card[data-color="green"]::before  { background: linear-gradient(90deg, #34d399, #06b6d4); }

.metric-card:hover { border-color: rgba(255, 255, 255, 0.13); }

.metric-label {
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #52525b;
  margin-bottom: 0.4rem;
}

.metric-value {
  font-family: var(--heading);
  font-size: 2rem;
  font-weight: 700;
  color: #fff;
  line-height: 1;
}

.metric-unit { font-size: 0.9rem; color: #71717a; font-weight: 400; margin-left: 0.2rem; }

.metric-sub { font-size: 0.62rem; color: #52525b; margin-top: 0.35rem; }

.metric-badge {
  display: inline-block;
  font-size: 0.55rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.15rem 0.5rem;
  border-radius: 100px;
  margin-top: 0.35rem;
}

.badge-ok   { background: rgba(52, 211, 153, 0.15); color: #34d399; }
.badge-warn { background: rgba(251, 191, 36, 0.15); color: #fbbf24; }
.badge-high { background: rgba(239, 68, 68, 0.15); color: #f87171; }

.results-main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 320px;
  gap: 1rem;
  margin-bottom: 1rem;
}

@media (max-width: 1100px) {
  .results-main-grid { grid-template-columns: 1fr 1fr; }
}

.panel-title-bar {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #52525b;
  margin-bottom: 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.panel-title-bar::before {
  content: '';
  width: 3px;
  height: 12px;
  background: linear-gradient(180deg, #a855f7, #fbbf24);
  border-radius: 2px;
  flex-shrink: 0;
}

.vad-bar-row { margin-bottom: 1rem; }

.vad-bar-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.4rem;
}

.vad-bar-name { font-size: 0.75rem; color: #a1a1aa; }

.vad-bar-scores { display: flex; gap: 0.75rem; }

.vad-score-true { font-size: 0.7rem; color: #22d3ee; font-weight: 700; }
.vad-score-pred { font-size: 0.7rem; color: #a855f7; font-weight: 700; }
.vad-score-label { font-size: 0.6rem; color: #52525b; margin-left: 0.2rem; }

.vad-track {
  height: 6px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 100px;
  position: relative;
}

.vad-fill {
  position: absolute;
  top: 0; left: 0;
  height: 100%;
  border-radius: 100px;
}

.ratio-chips {
  display: flex;
  gap: 0.75rem;
  margin-top: 1rem;
}

.ratio-chip {
  flex: 1;
  padding: 0.75rem;
  border-radius: 10px;
  text-align: center;
}

.ratio-chip[data-color="purple"] {
  background: rgba(168, 85, 247, 0.06);
  border: 1px solid rgba(168, 85, 247, 0.15);
}

.ratio-chip[data-color="yellow"] {
  background: rgba(251, 191, 36, 0.06);
  border: 1px solid rgba(251, 191, 36, 0.15);
}

.ratio-chip-label { font-size: 0.55rem; color: #52525b; text-transform: uppercase; margin-bottom: 0.25rem; }

.ratio-chip-val {
  font-family: var(--heading);
  font-size: 1.3rem;
  font-weight: 700;
}

.ratio-chip[data-color="purple"] .ratio-chip-val { color: #a855f7; }
.ratio-chip[data-color="yellow"] .ratio-chip-val { color: #fbbf24; }

.channel-bar-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.3rem;
}

.channel-bar-name {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: #52525b;
  width: 30px;
}

.channel-bar-track {
  flex: 1;
  height: 5px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 100px;
}

.channel-bar-fill { height: 100%; border-radius: 100px; opacity: 0.8; }

.channel-bar-val {
  font-family: var(--mono);
  font-size: 0.58rem;
  width: 30px;
  text-align: right;
}

.results-bottom-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

@media (max-width: 800px) {
  .results-bottom-grid { grid-template-columns: 1fr; }
}

.screening-score-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.screening-score-label { font-size: 0.8rem; color: #a1a1aa; }

.screening-score-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.screening-bar-wrap { width: 80px; }

.screening-bar-track {
  height: 4px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 100px;
}

.screening-bar-fill { height: 100%; border-radius: 100px; }

.screening-score-num {
  font-size: 0.8rem;
  font-weight: 700;
  color: #fff;
  width: 36px;
  text-align: right;
}

.screening-score-unit { font-size: 0.6rem; color: #52525b; }

.screening-rec-box {
  margin-top: 1rem;
  padding: 0.75rem;
  background: rgba(251, 191, 36, 0.05);
  border: 1px solid rgba(251, 191, 36, 0.15);
  border-radius: 10px;
}

.screening-rec-title {
  font-size: 0.65rem;
  color: #fbbf24;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.screening-rec-text { font-size: 0.75rem; color: #a1a1aa; line-height: 1.5; }

.screening-disclaimer {
  font-size: 0.65rem;
  color: #3f3f46;
  margin-top: 0.75rem;
  font-style: italic;
}

.explanation-para {
  font-size: 0.85rem;
  line-height: 1.7;
  color: #a1a1aa;
  margin-bottom: 1.25rem;
  padding: 1rem;
  background: rgba(168, 85, 247, 0.04);
  border: 1px solid rgba(168, 85, 247, 0.1);
  border-radius: 12px;
}

.explanation-para strong { color: #fbbf24; }

.top-feat-row {
  display: flex;
  align-items: center;
  padding: 0.4rem 0;
  font-size: 0.75rem;
  gap: 0.75rem;
}

.feat-name { color: #71717a; font-family: var(--mono); flex-shrink: 0; width: 120px; }

.feat-bar-wrap { flex: 1; }

.feat-bar-track {
  height: 3px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 100px;
}

.feat-bar-fill { height: 100%; border-radius: 100px; }

.feat-val {
  font-size: 0.65rem;
  font-family: var(--mono);
  width: 54px;
  text-align: right;
  flex-shrink: 0;
}

.feat-val.positive { color: #fbbf24; }
.feat-val.negative { color: #71717a; }
```

- [ ] **Step 2: Verify no build errors**

```bash
cd backend/frontend && npm run build 2>&1 | tail -5
```

Expected: `✓ built in Xs`

- [ ] **Step 3: Commit**

```bash
git add backend/frontend/src/App.css
git commit -m "feat: add CSS for ConnectScreen, ScanScreen, ResultsScreen"
```

---

## Task 7: ConnectScreen Component

**Files:**
- Create: `backend/frontend/src/components/ConnectScreen.tsx`

- [ ] **Step 1: Create the component**

```tsx
// src/components/ConnectScreen.tsx
import { useCallback, useEffect, useState } from 'react'
import type { DemoPatient } from '../types'

const REC_LABELS: Record<string, string> = {
  no_concern: 'No concerns',
  monitor: 'Monitor',
  consult_pcp: 'Consult PCP',
  seek_specialist: 'See specialist',
}

interface Props {
  onStart: (patient: DemoPatient) => void
}

export function ConnectScreen({ onStart }: Props) {
  const [patients, setPatients] = useState<DemoPatient[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detecting, setDetecting] = useState(true)

  useEffect(() => {
    fetch('/api/demo-patients')
      .then((r) => r.json())
      .then((data: DemoPatient[]) => setPatients(data))
      .catch(() => {/* API unreachable — start button stays disabled */})

    const timer = setTimeout(() => setDetecting(false), 2500)
    return () => clearTimeout(timer)
  }, [])

  const handleStart = useCallback(() => {
    const patient = patients.find((p) => p.id === selected)
    if (patient) onStart(patient)
  }, [patients, selected, onStart])

  const selectedPatient = patients.find((p) => p.id === selected) ?? null

  return (
    <div className="connect-screen">
      <nav className="connect-nav">
        <div className="nav-logo">
          <div className="nav-logo-icon">●</div> STRAIN
        </div>
        <div className={`status-pill ${detecting ? 'searching' : 'found'}`}>
          <div className="status-dot" />
          {detecting ? 'Searching for device…' : 'Device Found'}
        </div>
      </nav>

      <main className="connect-main">
        <div className="connect-hero-text">
          <h1>Connect your <span>EEG headset</span></h1>
          <p>
            Place the EMOTIV EPOC+ headset on the subject. STRAIN will detect the
            device automatically and begin neural calibration.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
          <div className="device-zone">
            <div className="device-rings">
              <div className="device-ring" />
              <div className="device-ring" />
              <div className="device-ring" />
            </div>
            <div className="device-icon-box">🧠</div>
          </div>
          <div className="scan-label">
            {detecting ? '● Scanning for device on port COM3…' : '● 14 channels active · 128 Hz'}
          </div>
        </div>

        <div className="patient-section">
          <div className="patient-section-label">— Select patient profile —</div>
          <div className="patient-cards-row">
            {patients.map((p) => (
              <div
                key={p.id}
                className={`patient-card${selected === p.id ? ' selected' : ''}`}
                data-accent={p.accent}
                onClick={() => setSelected(p.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && setSelected(p.id)}
              >
                <div className="patient-avatar-box">{p.avatar}</div>
                <div className="patient-card-name">{p.name}</div>
                <div className="patient-card-tag">{p.tag}</div>
                <div className="patient-card-desc">{p.description}</div>
                <div className="patient-card-meta">
                  <span className="patient-chip">Epoch #{p.epoch_index}</span>
                  <span className="patient-chip">DREAMER</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="connect-cta">
          <button
            type="button"
            className={`cta-primary-btn${selectedPatient ? ' ready' : ''}`}
            onClick={handleStart}
            disabled={!selectedPatient}
          >
            Begin Neural Scan →
          </button>
          <div className="connect-footnote">
            {detecting
              ? 'Searching for EEG device…'
              : 'Device detected · 14 channels active · 128 Hz sampling'}
          </div>
        </div>
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript**

```bash
cd backend/frontend && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add backend/frontend/src/components/ConnectScreen.tsx
git commit -m "feat: add ConnectScreen component"
```

---

## Task 8: ScanScreen Component

**Files:**
- Create: `backend/frontend/src/components/ScanScreen.tsx`

- [ ] **Step 1: Create the component**

```tsx
// src/components/ScanScreen.tsx
import { useEffect, useRef, useState } from 'react'
import { Brain3D } from './Brain3D'
import type { DemoPatient, DreamerAnalyzeResponse } from '../types'

const CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] as const

const PHASES = [
  'Device handshake',
  'Signal quality check',
  'Epoch extraction (256 samples)',
  'Welch PSD · band features',
  'VAD regression',
  'Mental health screening',
  'Report generation',
] as const

const PHASE_TIMES_MS = [1000, 2000, 3500, 6000, 7500, 8500, 9000] as const

const SCAN_DURATION_MS = 9000

// Deterministic per-channel power values for the live feed display (seeded by epoch index)
function mockChannelPower(epochIndex: number): Record<string, number> {
  return Object.fromEntries(
    CHANNELS.map((ch, i) => [ch, 0.6 + ((epochIndex * 7 + i * 13) % 17) / 10])
  )
}

// Wave path for a channel mini-SVG (deterministic, no random)
function wavePath(seed: number, value: number): string {
  const pts = Array.from({ length: 20 }, (_, j) => {
    const y = 12 + Math.sin(j * 0.7 + seed * 0.9) * 5 * value + Math.sin(j * 1.4 + seed * 0.3) * 3
    return `${j * 10},${Math.max(2, Math.min(22, y))}`
  })
  return `M ${pts.join(' L ')}`
}

interface Props {
  patient: DemoPatient
  onComplete: (data: DreamerAnalyzeResponse) => void
}

export function ScanScreen({ patient, onComplete }: Props) {
  const [completedPhases, setCompletedPhases] = useState(0)
  const [progress, setProgress] = useState(0)
  const scanResult = useRef<DreamerAnalyzeResponse | null>(null)
  const timerDone = useRef(false)
  const apiDone = useRef(false)

  const mockPower = mockChannelPower(patient.epoch_index)

  // Attempt advance when both conditions met
  const tryAdvance = () => {
    if (timerDone.current && apiDone.current && scanResult.current) {
      onComplete(scanResult.current)
    }
  }

  useEffect(() => {
    // Fire API call
    fetch('/api/analyze/dreamer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epoch_index: patient.epoch_index }),
    })
      .then((r) => {
        if (!r.ok) throw new Error('API error')
        return r.json() as Promise<DreamerAnalyzeResponse>
      })
      .then((data) => {
        scanResult.current = data
        apiDone.current = true
        tryAdvance()
      })
      .catch(() => {
        // On error, create a minimal placeholder so the demo still advances
        scanResult.current = {
          epoch_index: patient.epoch_index,
          subject_id: 0,
          trial_id: 0,
          true_vad: { valence: 3.0, arousal: 3.0, dominance: 3.0 },
          features: { spectral_ratios: { theta_alpha: 0.85, beta_alpha: 1.4 }, band_mean_power: {} },
          predicted_vad: null,
          explanation: { natural_language_explanation: 'API unreachable.', per_target_top_features: {} },
          screening: null,
        }
        apiDone.current = true
        tryAdvance()
      })

    // Phase timers
    const phaseTimers = PHASE_TIMES_MS.map((t, i) =>
      setTimeout(() => setCompletedPhases(i + 1), t)
    )

    // Progress bar update every 90ms
    const start = Date.now()
    const progressInterval = setInterval(() => {
      const pct = Math.min(100, ((Date.now() - start) / SCAN_DURATION_MS) * 100)
      setProgress(Math.round(pct))
    }, 90)

    // Main timer — advance after 9s
    const mainTimer = setTimeout(() => {
      timerDone.current = true
      tryAdvance()
    }, SCAN_DURATION_MS)

    return () => {
      phaseTimers.forEach(clearTimeout)
      clearTimeout(mainTimer)
      clearInterval(progressInterval)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient.epoch_index])

  const waveColor = (val: number) => (val > 1.8 ? '#fbbf24' : val < 0.85 ? '#34d399' : '#a855f7')
  const valClass = (val: number) => (val > 1.8 ? 'high' : val < 0.85 ? 'low' : 'mid')

  return (
    <div className="scan-screen">
      <nav className="scan-nav">
        <div className="nav-logo">
          <div className="nav-logo-icon">●</div> STRAIN
        </div>
        <div className="scan-patient-pill">
          <span className="spp-label">Patient:</span>
          <span className="spp-name">{patient.name}</span>
        </div>
      </nav>

      <div className="scan-body">
        {/* Left: Brain + status */}
        <div className="scan-brain-col">
          <Brain3D bandMeanPower={undefined} height="380px" />

          <div className="scan-status-block">
            <div className="scan-status-title">Analyzing neural patterns</div>
            <div className="scan-status-sub">
              ● Running Welch PSD · Ridge regression · VAD estimation
            </div>
            <div className="scan-progress-wrap">
              <div className="scan-progress-row">
                <span>Neural analysis</span>
                <span className="scan-progress-pct">{progress}%</span>
              </div>
              <div className="scan-progress-track">
                <div className="scan-progress-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          </div>
        </div>

        {/* Right: Channel feed */}
        <div className="scan-channel-col">
          <div className="scan-channel-header">
            <span>14-Channel EEG Feed</span>
            <span className="live-badge">● LIVE</span>
          </div>

          <div className="scan-channel-list">
            {CHANNELS.map((ch, i) => {
              const val = mockPower[ch] ?? 1.0
              const color = waveColor(val)
              const path = wavePath(i, val)
              return (
                <div key={ch} className="channel-row">
                  <div className="channel-name">{ch}</div>
                  <div className="channel-wave">
                    <svg viewBox="0 0 190 24" preserveAspectRatio="none" width="100%" height="100%">
                      <path d={path} fill="none" stroke={color} strokeWidth="1.5" opacity="0.7" />
                    </svg>
                  </div>
                  <div className={`channel-val ${valClass(val)}`}>{val.toFixed(2)}</div>
                </div>
              )
            })}
          </div>

          <div className="scan-phase-list">
            {PHASES.map((phase, i) => {
              const status =
                i < completedPhases ? 'done' : i === completedPhases ? 'active' : 'waiting'
              return (
                <div key={phase} className={`phase-item ${status}`}>
                  <div className="phase-dot" />
                  <span className="phase-label">{phase}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript**

```bash
cd backend/frontend && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add backend/frontend/src/components/ScanScreen.tsx
git commit -m "feat: add ScanScreen with 9s animated brain analysis"
```

---

## Task 9: ResultsScreen Component

**Files:**
- Create: `backend/frontend/src/components/ResultsScreen.tsx`

- [ ] **Step 1: Create the component**

```tsx
// src/components/ResultsScreen.tsx
import { useMemo } from 'react'
import { Brain3D } from './Brain3D'
import { MoodMeter } from './MoodMeter'
import type { DemoPatient, DreamerAnalyzeResponse } from '../types'

const REC_COPY: Record<string, string> = {
  no_concern: 'No neurological concerns identified in this scan window.',
  monitor: 'Elevated markers detected. Recommend follow-up assessment within 2 weeks.',
  consult_pcp: 'Significant patterns observed. Recommend consultation with a primary care physician.',
  seek_specialist: 'Notable neural stress markers. Recommend specialist referral.',
}

const CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] as const

interface Props {
  patient: DemoPatient
  scanData: DreamerAnalyzeResponse
  onReset: () => void
}

export function ResultsScreen({ patient, scanData, onReset }: Props) {
  const pred = scanData.predicted_vad
  const screening = scanData.screening
  const features = scanData.features
  const bandPower = features.band_mean_power

  const channelPowers = useMemo(() => {
    return CHANNELS.map((ch) => ({
      ch,
      val: bandPower[`beta_${ch}`] ?? bandPower[`theta_${ch}`] ?? 0.5,
    }))
  }, [bandPower])

  const maxPower = Math.max(...channelPowers.map((c) => c.val), 1e-6)

  const channelColor = (val: number) =>
    val > maxPower * 0.75 ? '#fbbf24' : val > maxPower * 0.4 ? '#a855f7' : '#34d399'

  const emotionLabel = useMemo(() => {
    if (!pred) return 'N/A'
    if (pred.valence >= 3.5) return 'POSITIVE'
    if (pred.valence <= 2.5) return 'NEGATIVE'
    return 'NEUTRAL'
  }, [pred])

  const badgeClass = (score: number) =>
    score > 70 ? 'badge-high' : score > 45 ? 'badge-warn' : 'badge-ok'

  const vadBadge = (v: number) =>
    v > 4 ? 'badge-high' : v > 2.5 ? 'badge-warn' : 'badge-ok'

  const topFeatures = useMemo(() => {
    const all = Object.values(scanData.explanation.per_target_top_features).flat()
    const seen = new Set<string>()
    const deduped: { name: string; contribution: number }[] = []
    for (const f of all) {
      if (!seen.has(f.name)) { seen.add(f.name); deduped.push(f) }
    }
    return deduped
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 6)
  }, [scanData.explanation.per_target_top_features])

  const maxContrib = Math.max(...topFeatures.map((f) => Math.abs(f.contribution)), 1e-6)

  const scanTime = new Date().toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className="results-screen">
      <nav className="results-nav">
        <div className="nav-logo">
          <div className="nav-logo-icon">●</div> STRAIN
        </div>
        <div className="results-nav-actions">
          <div className="result-badge">✓ Scan complete</div>
          <button
            type="button"
            className="result-action-btn"
            onClick={() => window.print()}
          >
            ⬇ Download Report
          </button>
          <button type="button" className="rescan-btn" onClick={onReset}>
            ↺ New Scan
          </button>
        </div>
      </nav>

      <div className="results-content">
        {/* Patient header */}
        <div className="patient-header">
          <div className="patient-header-left">
            <div className="patient-header-avatar">{patient.avatar}</div>
            <div>
              <div className="patient-header-name">{patient.name}</div>
              <div className="patient-header-sub">
                {patient.age} · {patient.profession} · Epoch #{scanData.epoch_index} · Subject S{String(scanData.subject_id).padStart(2, '0')} · Trial {scanData.trial_id}
              </div>
            </div>
          </div>
          <div className="patient-header-time">Scanned {scanTime} · 128 Hz · 14 channels · 2s epoch</div>
        </div>

        {/* Top metric row */}
        <div className="metric-row">
          <div className="metric-card" data-color="yellow">
            <div className="metric-label">Valence</div>
            <div className="metric-value">
              {pred ? pred.valence.toFixed(1) : '—'}
              <span className="metric-unit">/ 5</span>
            </div>
            <div className="metric-sub">Predicted by Ridge VAD</div>
            {pred && <div className={`metric-badge ${vadBadge(pred.valence)}`}>
              {pred.valence > 3.5 ? 'Positive affect' : pred.valence < 2.5 ? 'Negative affect' : 'Neutral affect'}
            </div>}
          </div>

          <div className="metric-card" data-color="purple">
            <div className="metric-label">Arousal</div>
            <div className="metric-value">
              {pred ? pred.arousal.toFixed(1) : '—'}
              <span className="metric-unit">/ 5</span>
            </div>
            <div className="metric-sub">Neural activation level</div>
            {pred && <div className={`metric-badge ${vadBadge(pred.arousal)}`}>
              {pred.arousal > 3.5 ? 'Elevated' : pred.arousal < 2.5 ? 'Low' : 'Moderate'}
            </div>}
          </div>

          <div className="metric-card" data-color="cyan">
            <div className="metric-label">Dominance</div>
            <div className="metric-value">
              {pred ? pred.dominance.toFixed(1) : '—'}
              <span className="metric-unit">/ 5</span>
            </div>
            <div className="metric-sub">Sense of control</div>
            {pred && <div className={`metric-badge ${vadBadge(pred.dominance)}`}>
              {pred.dominance > 3.5 ? 'In control' : pred.dominance < 2.5 ? 'Low control' : 'Moderate'}
            </div>}
          </div>

          <div className="metric-card" data-color="green">
            <div className="metric-label">Emotion State</div>
            <div className="metric-value" style={{ fontSize: '1.3rem', marginTop: '0.15rem' }}>
              {emotionLabel}
            </div>
            <div className="metric-sub">
              β/α: {features.spectral_ratios.beta_alpha.toFixed(3)}
            </div>
            <div className={`metric-badge ${emotionLabel === 'POSITIVE' ? 'badge-ok' : emotionLabel === 'NEGATIVE' ? 'badge-high' : 'badge-warn'}`}>
              θ/α: {features.spectral_ratios.theta_alpha.toFixed(3)}
            </div>
          </div>
        </div>

        {/* Main 3-col grid */}
        <div className="results-main-grid">
          {/* Brain3D */}
          <div className="panel">
            <div className="panel-title-bar">Live Brain Activity</div>
            <Brain3D bandMeanPower={Object.keys(bandPower).length > 0 ? bandPower : undefined} height="280px" />
          </div>

          {/* VAD + Mood */}
          <div className="panel">
            <div className="panel-title-bar">Valence · Arousal · Dominance</div>
            {pred ? (
              <>
                {(['valence', 'arousal', 'dominance'] as const).map((dim) => {
                  const trueVal = scanData.true_vad[dim]
                  const predVal = pred[dim]
                  return (
                    <div key={dim} className="vad-bar-row">
                      <div className="vad-bar-header">
                        <span className="vad-bar-name" style={{ textTransform: 'capitalize' }}>{dim}</span>
                        <div className="vad-bar-scores">
                          <span>
                            <span className="vad-score-true">{trueVal.toFixed(1)}</span>
                            <span className="vad-score-label">true</span>
                          </span>
                          <span>
                            <span className="vad-score-pred">{predVal.toFixed(1)}</span>
                            <span className="vad-score-label">pred</span>
                          </span>
                        </div>
                      </div>
                      <div className="vad-track">
                        <div className="vad-fill" style={{ width: `${(trueVal / 5) * 100}%`, background: '#22d3ee', opacity: 0.5 }} />
                        <div className="vad-fill" style={{ width: `${(predVal / 5) * 100}%`, background: '#a855f7' }} />
                      </div>
                    </div>
                  )
                })}

                <div style={{ marginTop: '1.25rem' }}>
                  <div className="panel-title-bar">Affective Space</div>
                  <MoodMeter
                    dataPoints={[
                      {
                        label: 'True',
                        valence: scanData.true_vad.valence,
                        arousal: scanData.true_vad.arousal,
                        minV: 1, maxV: 5, minA: 1, maxA: 5,
                        color: '#22d3ee',
                      },
                      {
                        label: 'Pred',
                        valence: pred.valence,
                        arousal: pred.arousal,
                        minV: 1, maxV: 5, minA: 1, maxA: 5,
                        color: '#a855f7',
                      },
                    ]}
                  />
                </div>
              </>
            ) : (
              <p style={{ color: '#52525b', fontSize: '0.8rem' }}>
                VAD model not trained. Run: <code>python scripts/train_dreamer_vad.py</code>
              </p>
            )}
          </div>

          {/* Channel activity */}
          <div className="panel">
            <div className="panel-title-bar">Channel Activity</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', marginBottom: '1.25rem' }}>
              {channelPowers.map(({ ch, val }) => (
                <div key={ch} className="channel-bar-row">
                  <div className="channel-bar-name">{ch}</div>
                  <div className="channel-bar-track">
                    <div
                      className="channel-bar-fill"
                      style={{
                        width: `${(val / maxPower) * 100}%`,
                        background: channelColor(val),
                      }}
                    />
                  </div>
                  <div className="channel-bar-val" style={{ color: channelColor(val) }}>
                    {val.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>

            <div className="panel-title-bar">Spectral Ratios</div>
            <div className="ratio-chips">
              <div className="ratio-chip" data-color="purple">
                <div className="ratio-chip-label">θ/α</div>
                <div className="ratio-chip-val">{features.spectral_ratios.theta_alpha.toFixed(3)}</div>
              </div>
              <div className="ratio-chip" data-color="yellow">
                <div className="ratio-chip-label">β/α</div>
                <div className="ratio-chip-val">{features.spectral_ratios.beta_alpha.toFixed(3)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom 2-col grid */}
        <div className="results-bottom-grid">
          {/* Screening */}
          <div className="panel">
            <div className="panel-title-bar">Mental Health Screening</div>
            {screening ? (
              <>
                {[
                  { label: 'Depression Risk', score: screening.depression_risk.score, gradient: 'linear-gradient(90deg,#fbbf24,#f97316)' },
                  { label: 'Anxiety Risk', score: screening.anxiety_risk.score, gradient: 'linear-gradient(90deg,#a855f7,#f97316)' },
                  { label: 'Cognitive Load', score: screening.cognitive_load.score, gradient: 'linear-gradient(90deg,#fbbf24,#a855f7)' },
                ].map(({ label, score, gradient }) => (
                  <div key={label} className="screening-score-row">
                    <span className="screening-score-label">{label}</span>
                    <div className="screening-score-right">
                      <div className="screening-bar-wrap">
                        <div className="screening-bar-track">
                          <div
                            className="screening-bar-fill"
                            style={{ width: `${score}%`, background: gradient }}
                          />
                        </div>
                      </div>
                      <div className="screening-score-num">
                        {Math.round(score)}<span className="screening-score-unit">/100</span>
                      </div>
                    </div>
                  </div>
                ))}
                <div className="screening-rec-box">
                  <div className="screening-rec-title">⚠ Recommendation</div>
                  <div className="screening-rec-text">
                    {REC_COPY[screening.recommendation] ?? screening.recommendation}
                  </div>
                </div>
                <div className="screening-disclaimer">{screening.disclaimer}</div>
              </>
            ) : (
              <p style={{ color: '#52525b', fontSize: '0.8rem' }}>
                No screening available — VAD model required.
              </p>
            )}
          </div>

          {/* AI Explanation */}
          <div className="panel">
            <div className="panel-title-bar">AI Explanation</div>
            <div className="explanation-para">
              {scanData.explanation.natural_language_explanation}
            </div>

            {topFeatures.length > 0 && (
              <>
                <div className="panel-title-bar">Top Contributing Features</div>
                {topFeatures.map((f) => {
                  const pct = (Math.abs(f.contribution) / maxContrib) * 100
                  const isPos = f.contribution >= 0
                  return (
                    <div key={f.name} className="top-feat-row">
                      <div className="feat-name">{f.name}</div>
                      <div className="feat-bar-wrap">
                        <div className="feat-bar-track">
                          <div
                            className="feat-bar-fill"
                            style={{
                              width: `${pct}%`,
                              background: isPos ? '#fbbf24' : 'rgba(168,85,247,0.6)',
                            }}
                          />
                        </div>
                      </div>
                      <div className={`feat-val ${isPos ? 'positive' : 'negative'}`}>
                        {isPos ? '+' : ''}{f.contribution.toFixed(4)}
                      </div>
                    </div>
                  )
                })}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript**

```bash
cd backend/frontend && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add backend/frontend/src/components/ResultsScreen.tsx
git commit -m "feat: add ResultsScreen with full analysis dashboard"
```

---

## Task 10: Rewrite App.tsx

**Files:**
- Modify: `backend/frontend/src/App.tsx`

- [ ] **Step 1: Replace entire `App.tsx` with the phase shell**

```tsx
// src/App.tsx
import { useState } from 'react'
import { ConnectScreen } from './components/ConnectScreen'
import { ScanScreen } from './components/ScanScreen'
import { ResultsScreen } from './components/ResultsScreen'
import type { DemoPatient, DreamerAnalyzeResponse, Phase } from './types'
import './App.css'

export default function App() {
  const [phase, setPhase] = useState<Phase>('connect')
  const [patient, setPatient] = useState<DemoPatient | null>(null)
  const [scanData, setScanData] = useState<DreamerAnalyzeResponse | null>(null)

  const handleStart = (p: DemoPatient) => {
    setPatient(p)
    setPhase('scanning')
  }

  const handleScanComplete = (data: DreamerAnalyzeResponse) => {
    setScanData(data)
    setPhase('results')
  }

  const handleReset = () => {
    setPatient(null)
    setScanData(null)
    setPhase('connect')
  }

  if (phase === 'scanning' && patient) {
    return <ScanScreen patient={patient} onComplete={handleScanComplete} />
  }

  if (phase === 'results' && patient && scanData) {
    return <ResultsScreen patient={patient} scanData={scanData} onReset={handleReset} />
  }

  return <ConnectScreen onStart={handleStart} />
}
```

- [ ] **Step 2: Verify TypeScript**

```bash
cd backend/frontend && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Full build check**

```bash
cd backend/frontend && npm run build 2>&1 | tail -10
```

Expected: `✓ built in Xs` with no errors

- [ ] **Step 4: Commit**

```bash
git add backend/frontend/src/App.tsx
git commit -m "feat: rewrite App.tsx as phase state machine (connect → scan → results)"
```

---

## Task 11: Smoke Test End-to-End

- [ ] **Step 1: Start the backend**

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Expected: `Application startup complete.`

- [ ] **Step 2: Start the frontend**

```bash
cd backend/frontend && npm run dev
```

Expected: `VITE ready on http://localhost:5173`

- [ ] **Step 3: Open http://localhost:5173 and verify**

Walk through the full flow:
1. **Connect screen:** Status pill shows `Searching for device…`, after 2.5s shows `Device Found`. Three patient cards load from the API. Clicking a card highlights it (purple border, slight lift). The "Begin Neural Scan" button becomes fully opaque.
2. **Scan screen:** Click "Begin Neural Scan". Full-screen brain loads with pulsing electrodes and dendrite lines. Channel list shows 14 channels with wave animations. Progress bar fills over 9 seconds. Phase tracker advances automatically. After 9s the screen transitions to Results.
3. **Results screen:** All 4 metric cards populated with real VAD values. Brain3D shows real band power. VAD bars show true vs predicted. Screening scores and AI explanation are displayed. "↺ New Scan" returns to Screen 1.

- [ ] **Step 4: Run backend tests**

```bash
pytest -q
```

Expected: all pass

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: brain monitor redesign complete — 3-screen polish"
```
