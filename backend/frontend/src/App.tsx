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

export default function App() {
  const [rowIndex, setRowIndex] = useState(0)
  const [csvData, setCsvData] = useState<CsvAnalyzeResponse | null>(null)
  const [csvMeta, setCsvMeta] = useState<{ n_samples?: number } | null>(null)
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

  useEffect(() => {
    void loadCsvMeta()
  }, [loadCsvMeta])

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

  useEffect(() => {
    void analyzeCsv(rowIndex)
  }, [rowIndex, analyzeCsv])

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

  return (
    <div className="emotiscan">
      <header className="hdr">
        <h1>EmotiScan v2.0</h1>
        <p className="sub">
          Prototype emotion / affect views — Kaggle tabular CSV. Not for
          clinical use.
        </p>
      </header>

      {csvMeta?.n_samples != null && (
        <p className="meta">Dataset rows: {csvMeta.n_samples}</p>
      )}

      <section className="controls">
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
        <button
          type="button"
          onClick={() => void analyzeCsv(rowIndex)}
          disabled={loading}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </section>

      {error && <div className="err">{error}</div>}

      {csvData && probChart && (
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
    </div>
  )
}
