// src/App.tsx
import { useEffect, useState } from 'react'
import { ConnectScreen } from './components/ConnectScreen'
import { ScanScreen } from './components/ScanScreen'
import { ResultsScreen } from './components/ResultsScreen'
import type { DemoPatient, DreamerAnalyzeResponse, Phase } from './types'
import './App.css'

export default function App() {
  const [phase, setPhase] = useState<Phase>('connect')
  const [patient, setPatient] = useState<DemoPatient | null>(null)
  const [scanData, setScanData] = useState<DreamerAnalyzeResponse | null>(null)

  // Deep-link: ?patient=alex-chen auto-starts scan for that patient
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('patient')
    if (!id) return
    fetch('/api/demo-patients')
      .then((r) => { if (!r.ok) throw new Error('not ok'); return r.json() })
      .then((data: unknown) => {
        if (!Array.isArray(data)) return
        const match = (data as DemoPatient[]).find((p) => p.id === id)
        if (match) { setPatient(match); setPhase('scanning') }
      })
      .catch(() => {})
  }, [])

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
