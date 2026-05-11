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
