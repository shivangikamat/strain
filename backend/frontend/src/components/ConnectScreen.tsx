// src/components/ConnectScreen.tsx
import { useCallback, useEffect, useState } from 'react'
import type { DemoPatient } from '../types'

interface Props {
  onStart: (patient: DemoPatient) => void
}

export function ConnectScreen({ onStart }: Props) {
  const [patients, setPatients] = useState<DemoPatient[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detecting, setDetecting] = useState(true)

  useEffect(() => {
    fetch('/api/demo-patients')
      .then((r) => { if (!r.ok) throw new Error('not ok'); return r.json() })
      .then((data: unknown) => { if (Array.isArray(data)) setPatients(data as DemoPatient[]) })
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
