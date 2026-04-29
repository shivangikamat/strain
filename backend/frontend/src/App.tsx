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
    <div className="emotiscan">
      <header className="hdr">
        <h1>EmotiScan v2.0</h1>
        <p className="sub">
          Prototype emotion / affect views — Kaggle tabular CSV or DREAMER EEG epochs. Not for
          clinical use.
        </p>
      </header>

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

      {mode === 'csv' && csvMeta?.n_samples != null && (
        <p className="meta">Dataset rows: {csvMeta.n_samples}</p>
      )}
      {mode === 'dreamer' && dreamerMeta?.n_epochs != null && (
        <p className="meta">DREAMER exported epochs: {dreamerMeta.n_epochs}</p>
      )}
      {mode === 'dreamer' && dreamerMeta == null && (
        <p className="meta warn">
          No DREAMER manifest — export epochs first (see README), then refresh.
        </p>
      )}

      <section className="controls">
        {mode === 'csv' ? (
          <label>
            Row index{' '}
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
            Epoch index{' '}
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
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </section>

      {error && <div className="err">{error}</div>}

      {mode === 'csv' && csvData && probChart && (
        <>
          <section className="panel grid2">
            <div>
              <h2>Emotion probabilities</h2>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={probChart} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <YAxis dataKey="name" type="category" width={80} />
                    <Tooltip formatter={(v: number) => [(v * 100).toFixed(1) + '%', 'prob']} />
                    <Bar dataKey="p" name="Probability" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="truth" style={{ marginTop: '1rem' }}>
                Ground truth label: <strong>{csvData.ground_truth_label}</strong> · Predicted:{' '}
                <strong>{csvData.analysis.classification.discrete_emotion}</strong> (
                {(csvData.analysis.classification.confidence * 100).toFixed(1)}% confidence)
              </p>
            </div>

            <div>
              <h2>Mood Meter</h2>
              <div style={{ marginTop: '1rem' }}>
                <MoodMeter 
                  dataPoints={[
                    {
                      label: 'Predicted',
                      valence: csvData.analysis.classification.valence,
                      arousal: csvData.analysis.classification.arousal,
                      minV: -1.0, maxV: 1.0, minA: 0.0, maxA: 1.0,
                      color: '#60a5fa'
                    }
                  ]} 
                />
              </div>
            </div>
          </section>

          <section className="panel grid2">
            <div>
              <h2>Demo screening (non-clinical)</h2>
              <ul className="scores">
                <li>
                  Depression risk (demo):{' '}
                  <strong>{csvData.analysis.screening.depression_risk.score.toFixed(0)}</strong> / 100
                </li>
                <li>
                  Anxiety risk (demo):{' '}
                  <strong>{csvData.analysis.screening.anxiety_risk.score.toFixed(0)}</strong> / 100
                </li>
                <li>
                  Recommendation: <strong>{csvData.analysis.screening.recommendation}</strong>
                </li>
              </ul>
              <p className="disclaimer">{csvData.analysis.screening.disclaimer}</p>
            </div>
            <div className="brain3d-col">
              <h2>Live Brain Activity</h2>
              <Brain3D bandMeanPower={csvBrainBandPower} />
            </div>
          </section>

          <section className="panel">
            <h2>Explanation</h2>
            <p>{csvData.analysis.explanation.natural_language_explanation}</p>
            <h3>Top linear coefficients × input</h3>
            <ul className="topf">
              {csvData.analysis.explanation.top_features.slice(0, 8).map((t) => (
                <li key={t.name}>
                  <code>{t.name}</code> — {t.contribution >= 0 ? '+' : ''}
                  {t.contribution.toFixed(4)}
                </li>
              ))}
            </ul>
          </section>
        </>
      )}

      {mode === 'dreamer' && dreamerData && (
        <>
          <section className="panel grid2">
            <div>
              <h2>DREAMER epoch</h2>
              <p className="truth">
                Epoch <strong>{dreamerData.epoch_index}</strong> · subject{' '}
                <strong>{dreamerData.subject_id}</strong> · trial <strong>{dreamerData.trial_id}</strong>
              </p>
              {dreamerData.predicted_vad != null && vadCompare && (
                <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                  <div style={{ flex: '1 1 auto', minWidth: 0 }}>
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
                  <div style={{ flex: '0 0 200px' }}>
                    <h3 style={{ textAlign: 'center' }}>Mood Meter</h3>
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
              )}
            </div>
            <div className="brain3d-col">
              <h2>Live Brain Activity</h2>
              <Brain3D bandMeanPower={dreamerData.features.band_mean_power} />
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

          <section className="panel grid2">
            <div>
              <h2>Welch band means (epoch)</h2>
              <p>
                θ/α: {dreamerData.features.spectral_ratios.theta_alpha.toFixed(3)} · β/α:{' '}
                {dreamerData.features.spectral_ratios.beta_alpha.toFixed(3)}
              </p>
              <pre className="bands">
                {JSON.stringify(dreamerData.features.band_mean_power, null, 2)}
              </pre>
            </div>
            {dreamerData.screening && (
              <div>
                <h2>Demo screening from VAD</h2>
                <ul className="scores">
                  <li>
                    Depression risk (demo):{' '}
                    <strong>{dreamerData.screening.depression_risk.score.toFixed(0)}</strong> / 100
                  </li>
                  <li>
                    Anxiety risk (demo):{' '}
                    <strong>{dreamerData.screening.anxiety_risk.score.toFixed(0)}</strong> / 100
                  </li>
                  <li>
                    Recommendation: <strong>{dreamerData.screening.recommendation}</strong>
                  </li>
                </ul>
                <p className="disclaimer">{dreamerData.screening.disclaimer}</p>
              </div>
            )}
          </section>

          <section className="panel">
            <h2>Explanation</h2>
            <p>{dreamerData.explanation.natural_language_explanation}</p>
            {Object.entries(dreamerData.explanation.per_target_top_features || {}).map(
              ([target, rows]) => (
                <div key={target}>
                  <h3>{target}</h3>
                  <ul className="topf">
                    {(rows as { name: string; contribution: number }[])
                      .slice(0, 6)
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
          </section>
        </>
      )}
    </div>
  )
}