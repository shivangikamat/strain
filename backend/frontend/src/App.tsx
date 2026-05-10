import { useCallback, useEffect, useMemo, useState } from 'react'
import { Brain3D } from './components/Brain3D'
import { MoodMeter } from './components/MoodMeter'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type CsvAnalyzeResponse = {
  row_index: number
  ground_truth_label: string
  analysis: {
    features: {
      spectral_ratios: { theta_alpha: number; beta_alpha: number }
      band_energy_proxy: Record<string, number>
    }
    classification: {
      discrete_emotion: string
      confidence: number
      probabilities: Record<string, number>
      valence: number
      arousal: number
    }
    screening: {
      disclaimer: string
      depression_risk: { score: number }
      anxiety_risk: { score: number }
      recommendation: string
      key_findings: string[]
    }
    explanation: {
      natural_language_explanation: string
      top_features: { name: string; contribution: number }[]
    }
  }
}

type DreamerAnalyzeResponse = {
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
    per_target_top_features: Record<
      string,
      { name: string; contribution: number }[]
    >
  }
  screening: {
    disclaimer: string
    depression_risk: { score: number; note?: string }
    anxiety_risk: { score: number; note?: string }
    recommendation: string
    key_findings: string[]
  } | null
}

type DataMode = 'csv' | 'dreamer'

export default function App() {
  const [mode, setMode] = useState<DataMode>('csv')
  const [rowIndex, setRowIndex] = useState(0)
  const [epochIndex, setEpochIndex] = useState(0)
  const [csvData, setCsvData] = useState<CsvAnalyzeResponse | null>(null)
  const [dreamerData, setDreamerData] = useState<DreamerAnalyzeResponse | null>(null)
  const [csvMeta, setCsvMeta] = useState<{ n_samples?: number } | null>(null)
  const [dreamerMeta, setDreamerMeta] = useState<{ n_epochs?: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const loadCsvMeta = useCallback(async () => {
    try {
      const r = await fetch('/api/dataset/meta')
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setCsvMeta({ n_samples: j.n_samples })
    } catch {
      setCsvMeta(null)
    }
  }, [])

  const loadDreamerMeta = useCallback(async () => {
    try {
      const r = await fetch('/api/dataset/dreamer/meta')
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setDreamerMeta({ n_epochs: j.n_epochs })
    } catch {
      setDreamerMeta(null)
    }
  }, [])

  useEffect(() => {
    void loadCsvMeta() 
    void loadDreamerMeta()
  }, [loadCsvMeta, loadDreamerMeta])

  const analyzeCsv = useCallback(async (idx: number) => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row_index: idx }),
      })
      if (!r.ok) throw new Error(await r.text())
      setCsvData(await r.json())
      setDreamerData(null)
    } catch (e) {
      setCsvData(null)
      setError(
        e instanceof Error
          ? e.message
          : 'API unreachable. Start backend: uvicorn api.main:app --reload --port 8000',
      )
    } finally {
      setLoading(false)
    }
  }, [])

  const analyzeDreamer = useCallback(async (idx: number) => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/analyze/dreamer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epoch_index: idx }),
      })
      if (!r.ok) throw new Error(await r.text())
      setDreamerData(await r.json())
      setCsvData(null)
    } catch (e) {
      setDreamerData(null)
      setError(
        e instanceof Error
          ? e.message
          : 'DREAMER export or manifest missing, or API down. See README.',
      )
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (mode === 'csv') void analyzeCsv(rowIndex)
    else void analyzeDreamer(epochIndex)
  }, [mode, rowIndex, epochIndex, analyzeCsv, analyzeDreamer])

  const probChart =
    csvData &&
    Object.entries(csvData.analysis.classification.probabilities).map(([name, value]) => ({
      name,
      p: value,
    }))

  const csvBrainBandPower = useMemo(() => {
    if (!csvData) return undefined
    const b = csvData.analysis.features.spectral_ratios.beta_alpha || 1
    const channels = [
      'AF3',
      'F7',
      'F3',
      'FC5',
      'T7',
      'P7',
      'O1',
      'O2',
      'P8',
      'T8',
      'FC6',
      'F4',
      'F8',
      'AF4',
    ] as const
    return Object.fromEntries(
      channels.map((ch, i) => {
        const spread = 0.75 + (i % 5) * 0.06
        return [`beta_${ch}`, b * spread] as const
      }),
    )
  }, [csvData])

  const vadCompare =
    dreamerData &&
    dreamerData.predicted_vad &&
    [
      {
        name: 'Valence',
        true: dreamerData.true_vad.valence,
        pred: dreamerData.predicted_vad.valence,
      },
      {
        name: 'Arousal',
        true: dreamerData.true_vad.arousal,
        pred: dreamerData.predicted_vad.arousal,
      },
      {
        name: 'Dominance',
        true: dreamerData.true_vad.dominance,
        pred: dreamerData.predicted_vad.dominance,
      },
    ]

  return (
    <div className="strain">
      {/* Navigation */}
      <nav className="top-nav">
        <div className="nav-logo">
          <div className="nav-logo-icon">●</div> STRAIN
        </div>
        <div className="nav-links">
          <a href="#">COURSES</a>
          <a href="#">ABOUT US</a>
          <a href="#">BLOG</a>
          <a href="#">PROJECTS</a>
        </div>
        <button className="nav-btn">GET THE APP</button>
      </nav>

      {/* Hero Section */}
      <div className="hero">
        <div className="lines-container">
          <svg xmlns="http://www.w3.org/2000/svg">
            <path d="M0,100 Q400,300 800,100 T1600,100 M0,300 Q400,100 800,300 T1600,300" />
            <path d="M0,500 Q400,700 800,500 T1600,500" />
            <path d="M400,0 L400,800 M1200,0 L1200,800" />
          </svg>
        </div>

        {/* Floating Nodes */}
        <div className="node n-top-left">
          <div className="node-icon">✨</div>
          <div className="node-content">
            <span className="node-label">Live Inference</span>
            <span className="node-value">It's easy</span>
          </div>
        </div>
        <div className="node n-bottom-left">
          <div className="node-icon orange">🧠</div>
          <div className="node-content">
            <span className="node-label">Current Row</span>
            <span className="node-value">{mode === 'csv' ? rowIndex : epochIndex}</span>
          </div>
        </div>
        <div className="node n-top-right">
          <div className="node-icon yellow">⚡</div>
          <div className="node-content">
            <span className="node-label">Confidence</span>
            <span className="node-value">I got it!</span>
          </div>
        </div>
        <div className="node n-bottom-right">
          <div className="node-icon">👤</div>
          <div className="node-content">
            <span className="node-label">Users</span>
            <span className="node-value">More than 1,000</span>
          </div>
        </div>

        <h1 className="hero-title">
          EMOTION <div className="hero-icon">👁️</div> SYSTEM
        </h1>
        <p className="hero-subtitle">
          DESIGN SYSTEMS FOR ENTERPRISES<br/>
          Prototype emotion classification and non-clinical demo screening from tabular EEG features or DREAMER epochs.
        </p>

        <div className="hero-controls">
          <div className="mode-toggle" role="tablist" aria-label="Data source">
            <button
              type="button"
              className={mode === 'csv' ? 'active' : ''}
              onClick={() => setMode('csv')}
            >
              Kaggle CSV
            </button>
            <button
              type="button"
              className={mode === 'dreamer' ? 'active' : ''}
              onClick={() => setMode('dreamer')}
            >
              DREAMER epochs
            </button>
          </div>

          <div className="controls">
            {mode === 'csv' ? (
              <label>
                <input
                  type="number"
                  min={0}
                  max={Math.max(0, (csvMeta?.n_samples ?? 2132) - 1)}
                  value={rowIndex}
                  onChange={(e) => setRowIndex(Number(e.target.value) || 0)}
                />
              </label>
            ) : (
              <label>
                <input
                  type="number"
                  min={0}
                  max={Math.max(0, (dreamerMeta?.n_epochs ?? 1) - 1)}
                  value={epochIndex}
                  onChange={(e) => setEpochIndex(Number(e.target.value) || 0)}
                />
              </label>
            )}
            <button
              type="button"
              onClick={() => (mode === 'csv' ? void analyzeCsv(rowIndex) : void analyzeDreamer(epochIndex))}
              disabled={loading}
            >
              {loading ? 'LOADING...' : "LET'S START"}
            </button>
          </div>
          
          {mode === 'csv' && csvMeta?.n_samples != null && (
            <p className="meta truth" style={{margin: 0}}>Dataset rows: {csvMeta.n_samples}</p>
          )}
          {mode === 'dreamer' && dreamerMeta?.n_epochs != null && (
            <p className="meta truth" style={{margin: 0}}>DREAMER exported epochs: {dreamerMeta.n_epochs}</p>
          )}
          {mode === 'dreamer' && dreamerMeta == null && (
            <p className="meta warn" style={{margin: 0}}>
              No DREAMER manifest — export epochs first (see README).
            </p>
          )}
          {error && <div className="err" style={{margin: 0, marginTop: '1rem'}}>{error}</div>}
        </div>
      </div>

      <div className="dashboard-content">
        {mode === 'csv' && csvData && probChart && (
          <section className="panel" style={{ position: 'relative' }}>
            <button className="nav-btn" style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', padding: '0.5rem', minWidth: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center' }} title="Download Report">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            </button>
            <div className="grid3">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div>
                  <h2>Emotion probabilities</h2>
                  <p className="truth" style={{ marginBottom: '1rem' }}>
                    Ground truth label: <strong>{csvData.ground_truth_label}</strong> · Predicted:{' '}
                    <strong>{csvData.analysis.classification.discrete_emotion}</strong> (
                    {(csvData.analysis.classification.confidence * 100).toFixed(1)}% confidence)
                  </p>
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={probChart} layout="vertical" margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                        <YAxis dataKey="name" type="category" width={80} />
                        <Tooltip formatter={(v: number) => [(v * 100).toFixed(1) + '%', 'prob']} />
                        <Bar dataKey="p" name="Probability" fill="#a855f7" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <h2>Mood Meter</h2>
                <div style={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <MoodMeter 
                    dataPoints={[
                      {
                        label: 'Predicted',
                        valence: csvData.analysis.classification.valence,
                        arousal: csvData.analysis.classification.arousal,
                        minV: -1.0, maxV: 1.0, minA: 0.0, maxA: 1.0,
                        color: '#a855f7'
                      }
                    ]} 
                  />
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <h2>Live Brain Activity</h2>
                <div className="brain3d-col" style={{ marginTop: '1rem', flexGrow: 1 }}>
                  <Brain3D bandMeanPower={csvBrainBandPower} />
                </div>
              </div>
            </div>
          </section>
        )}

        {mode === 'dreamer' && dreamerData && (
          <>
            <section className="panel" style={{ position: 'relative' }}>
              <button className="nav-btn" style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', padding: '0.5rem', minWidth: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center' }} title="Download Report">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              </button>
              <div className="grid3">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div>
                    <h2>DREAMER epoch</h2>
                    <p className="truth">
                      Epoch <strong>{dreamerData.epoch_index}</strong> · subject{' '}
                      <strong>{dreamerData.subject_id}</strong> · trial <strong>{dreamerData.trial_id}</strong>
                    </p>
                  </div>
                  
                  {dreamerData.predicted_vad != null && vadCompare && (
                    <>
                      <div>
                        <h3>VAD (1–5 scale)</h3>
                        <div className="chart-wrap" style={{ marginTop: '0.5rem' }}>
                          <ResponsiveContainer width="100%" height={180}>
                            <BarChart data={vadCompare}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis domain={[0, 5.5]} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="true" name="True" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                              <Bar dataKey="pred" name="Predicted" fill="#a855f7" radius={[4, 4, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {dreamerData.predicted_vad != null && vadCompare ? (
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <h2>Mood Meter</h2>
                    <div style={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <MoodMeter
                        dataPoints={[
                          {
                            label: 'True',
                            valence: dreamerData.true_vad.valence,
                            arousal: dreamerData.true_vad.arousal,
                            minV: 1,
                            maxV: 5,
                            minA: 1,
                            maxA: 5,
                            color: '#0ea5e9',
                          },
                          {
                            label: 'Pred',
                            valence: dreamerData.predicted_vad.valence,
                            arousal: dreamerData.predicted_vad.arousal,
                            minV: 1,
                            maxV: 5,
                            minA: 1,
                            maxA: 5,
                            color: '#a855f7',
                          },
                        ]}
                      />
                    </div>
                  </div>
                ) : (
                  <div />
                )}

                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <h2>Live Brain Activity</h2>
                  <div className="brain3d-col" style={{ marginTop: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)', flexGrow: 1 }}>
                    <Brain3D bandMeanPower={dreamerData.features.band_mean_power} />
                  </div>
                </div>
              </div>
            </section>

            {!dreamerData.predicted_vad && (
              <section className="panel warn-panel">
                <p>
                  VAD model not found. Train with:{' '}
                  <code>POST /api/internal/train-dreamer-vad</code> (after exporting DREAMER epochs).
                </p>
              </section>
            )}
          </>
        )}
      </div>

      {/* The White Section at the bottom */}
      <div className="white-section">
        <div className="white-section-inner">
          <div className="grid2">
            <div className="panel" style={{gridColumn: '1 / -1', background: 'transparent', border: 'none', boxShadow: 'none'}}>
              <h2 style={{fontSize: '3rem', margin: 0, lineHeight: '1.1'}}>The ultimate library<br/>for your products</h2>
            </div>
            {mode === 'csv' && csvData && (
              <>
                <div className="panel">
                  <h2>Demo screening (non-clinical)</h2>
                  <ul className="scores">
                    <li>
                      Depression risk: <strong>{csvData.analysis.screening.depression_risk.score.toFixed(0)}</strong> / 100
                    </li>
                    <li>
                      Anxiety risk: <strong>{csvData.analysis.screening.anxiety_risk.score.toFixed(0)}</strong> / 100
                    </li>
                    <li>
                      Recommendation: <strong>{csvData.analysis.screening.recommendation}</strong>
                    </li>
                  </ul>
                  <p className="disclaimer">{csvData.analysis.screening.disclaimer}</p>
                </div>

                <div className="panel">
                  <h2>Explanation</h2>
                  <p>{csvData.analysis.explanation.natural_language_explanation}</p>
                  <h3>Top linear coefficients × input</h3>
                  <ul className="topf">
                    {csvData.analysis.explanation.top_features.slice(0, 5).map((t) => (
                      <li key={t.name}>
                        <code>{t.name}</code> — {t.contribution >= 0 ? '+' : ''}
                        {t.contribution.toFixed(4)}
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            )}

            {mode === 'dreamer' && dreamerData && (
              <>
                <div className="panel">
                  <h2>Welch band means (epoch)</h2>
                  <p>
                    θ/α: {dreamerData.features.spectral_ratios.theta_alpha.toFixed(3)} · β/α:{' '}
                    {dreamerData.features.spectral_ratios.beta_alpha.toFixed(3)}
                  </p>

                  
                  {dreamerData.screening && (
                    <div style={{marginTop: '2rem'}}>
                      <h2>Demo screening from VAD</h2>
                      <ul className="scores">
                        <li>
                          Depression risk: <strong>{dreamerData.screening.depression_risk.score.toFixed(0)}</strong> / 100
                        </li>
                        <li>
                          Anxiety risk: <strong>{dreamerData.screening.anxiety_risk.score.toFixed(0)}</strong> / 100
                        </li>
                        <li>
                          Recommendation: <strong>{dreamerData.screening.recommendation}</strong>
                        </li>
                      </ul>
                      <p className="disclaimer">{dreamerData.screening.disclaimer}</p>
                    </div>
                  )}
                </div>

                <div className="panel">
                  <h2>Explanation</h2>
                  <p>{dreamerData.explanation.natural_language_explanation}</p>
                  {Object.entries(dreamerData.explanation.per_target_top_features || {}).map(
                    ([target, rows]) => (
                      <div key={target} style={{marginTop: '1rem'}}>
                        <h3>{target}</h3>
                        <ul className="topf">
                          {(rows as { name: string; contribution: number }[])
                            .slice(0, 4)
                            .map((t) => (
                              <li key={target + t.name}>
                                <code>{t.name}</code> — {t.contribution >= 0 ? '+' : ''}
                                {t.contribution.toFixed(4)}
                              </li>
                            ))}
                        </ul>
                      </div>
                    ),
                  )}
                </div>
              </>
            )}
            
            {!(mode === 'csv' && csvData) && !(mode === 'dreamer' && dreamerData) && (
              <div className="panel" style={{gridColumn: '1 / -1', textAlign: 'center'}}>
                <h2>The library is waiting...</h2>
                <p>Select a data source above and click "LET'S START" to populate the dashboard.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}