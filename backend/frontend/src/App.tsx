import { useCallback, useEffect, useState } from 'react'
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

type AnalyzeResponse = {
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
  const [data, setData] = useState<AnalyzeResponse | null>(null)
  const [meta, setMeta] = useState<{ n_samples?: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const loadMeta = useCallback(async () => {
    try {
      const r = await fetch('/api/dataset/meta')
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setMeta({ n_samples: j.n_samples })
    } catch {
      setMeta(null)
    }
  }, [])

  const analyze = useCallback(async (idx: number) => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row_index: idx }),
      })
      if (!r.ok) throw new Error(await r.text())
      setData(await r.json())
    } catch (e) {
      setData(null)
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
    void loadMeta()
  }, [loadMeta])

  useEffect(() => {
    void analyze(rowIndex)
  }, [rowIndex, analyze])

  const probChart =
    data &&
    Object.entries(data.analysis.classification.probabilities).map(
      ([name, value]) => ({
        name,
        p: value,
      }),
    )

  return (
    <div className="emotiscan">
      <header className="hdr">
        <h1>EmotiScan v2.0</h1>
        <p className="sub">
          Prototype emotion screen from Kaggle-derived features (
          <code>data/emotions.csv</code>). Not for clinical use.
        </p>
      </header>

      {meta?.n_samples != null && (
        <p className="meta">Dataset rows: {meta.n_samples}</p>
      )}

      <section className="controls">
        <label>
          Row index{' '}
          <input
            type="number"
            min={0}
            max={Math.max(0, (meta?.n_samples ?? 2132) - 1)}
            value={rowIndex}
            onChange={(e) => setRowIndex(Number(e.target.value) || 0)}
          />
        </label>
        <button type="button" onClick={() => void analyze(rowIndex)} disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </section>

      {error && <div className="err">{error}</div>}

      {data && probChart && (
        <>
          <section className="panel">
            <h2>Emotion probabilities</h2>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={probChart}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip formatter={(v: number) => [(v * 100).toFixed(1) + '%', 'prob']} />
                  <Legend />
                  <Bar dataKey="p" name="Probability" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="truth">
              Ground truth label: <strong>{data.ground_truth_label}</strong> · Predicted:{' '}
              <strong>{data.analysis.classification.discrete_emotion}</strong> (
              {(data.analysis.classification.confidence * 100).toFixed(1)}% confidence)
            </p>
          </section>

          <section className="panel grid2">
            <div>
              <h2>Demo screening (non-clinical)</h2>
              <ul className="scores">
                <li>
                  Depression risk (demo):{' '}
                  <strong>{data.analysis.screening.depression_risk.score.toFixed(0)}</strong> / 100
                </li>
                <li>
                  Anxiety risk (demo):{' '}
                  <strong>{data.analysis.screening.anxiety_risk.score.toFixed(0)}</strong> / 100
                </li>
                <li>
                  Recommendation: <strong>{data.analysis.screening.recommendation}</strong>
                </li>
              </ul>
              <p className="disclaimer">{data.analysis.screening.disclaimer}</p>
            </div>
            <div>
              <h2>Proxy spectral ratios</h2>
              <p>
                θ/α: {data.analysis.features.spectral_ratios.theta_alpha.toFixed(3)} · β/α:{' '}
                {data.analysis.features.spectral_ratios.beta_alpha.toFixed(3)}
              </p>
              <pre className="bands">
                {JSON.stringify(data.analysis.features.band_energy_proxy, null, 2)}
              </pre>
            </div>
          </section>

          <section className="panel">
            <h2>Explanation</h2>
            <p>{data.analysis.explanation.natural_language_explanation}</p>
            <h3>Top linear coefficients × input</h3>
            <ul className="topf">
              {data.analysis.explanation.top_features.slice(0, 8).map((t) => (
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
