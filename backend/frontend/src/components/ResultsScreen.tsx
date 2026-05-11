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
