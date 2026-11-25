import { useState } from 'react'
import api from '../api'

function percentage(value) {
  return value == null ? 'Not reported' : `${(value * 100).toFixed(1)}%`
}

export default function AskQuestion({ threshold, setThreshold }) {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function submit(event) {
    event.preventDefault()
    if (!question.trim()) {
      setError('Enter a question before asking.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await api.post('/ask', { question: question.trim(), threshold })
      setResult(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'The API could not process this question.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Ask a Question</h1>
        <p className="page-desc">Each configured provider receives the same question. Provider failures are shown, never replaced with fabricated answers.</p>
      </header>
      <div className="alert alert-warning">
        Trust is an estimate of correctness, not proof. Historical accuracy comes from ARC-Challenge; results on
        other topics are less certain. Free-form answer grouping uses normalized string matching, not semantic equivalence.
      </div>

      <section className="card ask-form-card">
        <form onSubmit={submit}>
          <label className="form-label" htmlFor="question">Your question</label>
          <textarea
            id="question"
            className="form-input question-input"
            value={question}
            onChange={event => setQuestion(event.target.value)}
            maxLength={4000}
            rows={5}
            placeholder="Ask a factual question…"
          />
          <div className="slider-container" style={{ margin: '18px 0' }}>
            <div className="slider-header">
              <label className="slider-label" htmlFor="ask-threshold">Trust threshold</label>
              <span className="slider-value">{threshold.toFixed(2)}</span>
            </div>
            <input id="ask-threshold" type="range" min="0" max="1" step="0.01" value={threshold}
              onChange={event => setThreshold(Number(event.target.value))} />
          </div>
          {error && <div className="alert alert-error" role="alert">{Array.isArray(error) ? error.map(item => item.msg).join(' ') : error}</div>}
          <button className="btn btn-primary btn-lg" type="submit" disabled={loading}>
            {loading ? 'Asking configured models…' : 'Ask TrustMeter'}
          </button>
        </form>
      </section>

      {loading && <div className="loading-container"><span className="loading-spinner" /><span>Waiting for provider responses…</span></div>}

      {result && (
        <section className="answer-results" aria-live="polite">
          <div className={`verdict-panel ${result.verdict ? `verdict-${result.verdict.toLowerCase()}` : ''}`}>
            <span className="verdict-label">{result.verdict || 'Trust estimator not trained'}</span>
            <div className="verdict-answer">
              {result.verdict === 'ABSTAIN'
                ? 'No model has a trustworthy answer.'
                : result.consensus_answer || 'No consensus answer was produced.'}
            </div>
            <p>{result.message}</p>
            <p>Models answered: <strong>{result.answered_model_count} of {result.configured_model_count}</strong></p>
            <p>Calibrated trust estimate: <strong>{percentage(result.trust_score)}</strong> · threshold {result.threshold.toFixed(2)}</p>
            {result.supporting_models.length > 0 && <p>Models supporting the consensus: {result.supporting_models.join(', ')}</p>}
            {result.dissenting_models.length > 0 && <p>Dissenting models: {result.dissenting_models.join(', ')}</p>}
          </div>
          <h2 className="section-title">Model responses</h2>
          <div className="grid-auto">
            {result.models.map(model => (
              <article className="card model-answer-card" key={model.model_id}>
                <div className="card-header">
                  <h3 className="card-title">{model.display_name}</h3>
                  {model.error
                    ? <span className="badge badge-danger">Error</span>
                    : <span className="badge badge-blue">Response received</span>}
                </div>
                {model.error
                  ? <p className="provider-error">{model.error}</p>
                  : <p className="model-answer-text">{model.answer}</p>}
                <div className="answer-meta">
                  <span>Self-reported confidence: {percentage(model.confidence == null ? null : model.confidence / 100)}</span>
                  <span>Latency: {model.latency_ms == null ? '—' : `${model.latency_ms.toFixed(0)} ms`}</span>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
