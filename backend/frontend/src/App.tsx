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
