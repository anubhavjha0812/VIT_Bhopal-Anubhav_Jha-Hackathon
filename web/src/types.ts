// Shapes of the FastAPI responses (src/riskengine/api.py).

export interface EngineConfig {
  sentiment: string
  events: string
  impact: string
  rebalancer: string
  shock: string
  impact_trigger: number
  window_start: string
  window_end: string
}

export interface Overview {
  config: EngineConfig
  engine_models: { sentiment: string; events: string }
  signal_models: string[]
  texts: number
  signals: number
  texts_by_source: Record<string, number>
  events: Record<string, number>
  alerts_naive: number // texts with impact above the trigger
  alerts: number // texts the Credit Chhanni gate passes to Module B
  trading_days: number
}

export interface ScoreRow {
  model: string
  n: number
  accuracy: number
  accuracy_ci95: [number, number]
  macro_f1: number
  macro_f1_ci95: [number, number]
  texts_per_s: number
  model_size_mb: number
}

export type Scoreboard = Record<string, ScoreRow[]>

export interface DailyRow {
  date: string
  ticker: string
  sentiment: number
  texts: number
  max_impact: number
}

export interface Signal {
  doc_id: string
  published_at: string
  source: string
  ticker: string
  role?: 'subject' | 'analyst' // missing in signal files made before DECISIONS E12
  text: string
  sentiment: number
  event: string
  analyst_rating?: boolean
  impact: number
  model: string
  fired?: boolean // Credit Chhanni passed it to Module B
  reason?: string
}

export interface PricePoint {
  date: string
  close: number
}

export interface Analysis {
  tickers: string[]
  analysts: string[]
  analyst_rating: boolean
  sentiment: number
  event: string
  impact: number
  model: string
}
