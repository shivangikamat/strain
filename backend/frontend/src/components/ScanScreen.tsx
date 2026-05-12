// src/components/ScanScreen.tsx
import { useEffect, useRef, useState } from 'react'
import { Brain3D } from './Brain3D'
import type { DemoPatient, DreamerAnalyzeResponse } from '../types'
import { apiUrl } from '../apiBase'

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
    fetch(apiUrl('/api/analyze/dreamer'), {
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
