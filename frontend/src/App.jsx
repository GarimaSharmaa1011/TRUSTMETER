
import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import api from './api'
import Navbar from './components/Navbar.jsx'

const Overview = lazy(() => import('./components/Overview.jsx'))
const AskQuestion = lazy(() => import('./components/AskQuestion.jsx'))
const BatchAsk = lazy(() => import('./components/BatchAsk.jsx'))
const Leaderboard = lazy(() => import('./components/Leaderboard.jsx'))
const ModelComparison = lazy(() => import('./components/ModelComparison.jsx'))
const ReliabilityAnalysis = lazy(() => import('./components/ReliabilityAnalysis.jsx'))
const ThresholdSettings = lazy(() => import('./components/ThresholdSettings.jsx'))

const PAGES = {
  overview: Overview,
  ask: AskQuestion,
  batch: BatchAsk,
  leaderboard: Leaderboard,
  comparison: ModelComparison,
  reliability: ReliabilityAnalysis,
  threshold: ThresholdSettings,
}

export default function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [threshold, setThreshold] = useState(0.7)
  const [thresholdInitialized, setThresholdInitialized] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [backendStatus, setBackendStatus] = useState(false)
  const [backendError, setBackendError] = useState(null)
  const [providers, setProviders] = useState([])
  const [evaluationError, setEvaluationError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const health = await api.get('/health')

      setBackendStatus(health.data.status === 'ok')
      setProviders(health.data.providers || [])
      setBackendError(null)
      setEvaluationError(health.data.evaluation_error)

      try {
        const result = await api.get('/metrics')
        setMetrics(result.data)

        if (
          !thresholdInitialized &&
          result.data.evaluation?.threshold != null
        ) {
          setThreshold(result.data.evaluation.threshold)
          setThresholdInitialized(true)
        }

        setEvaluationError(null)
      } catch (error) {
        setMetrics(null)
        setEvaluationError(
          error.response?.data?.detail ||
          'Saved evaluation metrics are unavailable.'
        )
      }
    } catch (error) {
      setBackendStatus(false)
      setProviders([])
      setMetrics(null)
      setBackendError(
        error.response?.data?.detail ||
        'Cannot reach the API. Start the backend and configure providers.'
      )
    }
  }, [thresholdInitialized])

  useEffect(() => {
    refresh()

    const timer = window.setInterval(refresh, 30000)

    return () => window.clearInterval(timer)
  }, [refresh])

  const Page = PAGES[activeTab] || Overview

  return (
    <div className="app-layout">
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        backendStatus={backendStatus}
      />

      <main className="main-content">
        {backendError && (
          <div className="alert alert-warning" role="status">
            {backendError}
          </div>
        )}

        <Suspense
          fallback={
            <div className="loading-container">
              <span className="loading-spinner" />
              Loading view…
            </div>
          }
        >
          <Page
            metrics={metrics}
            threshold={threshold}
            setThreshold={setThreshold}
            onMetricsRefresh={refresh}
            providers={providers}
            evaluationError={evaluationError}
          />
        </Suspense>
      </main>

      <footer className="app-footer">
        TrustMeter — When Should an AI Admit It Is Unsure? · Live verdicts
        estimate correctness; they are not proof.
      </footer>
    </div>
  )
}
