import { useState } from 'react'
import axios from 'axios'

function percent(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

export default function BatchAsk({ threshold }) {
  const [text, setText] = useState('')
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function submit(event) {
    event.preventDefault()
    const questions = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
    if (!questions.length) {
      setError('Enter at least one question, one per line.')
      return
    }
    if (questions.length > 20) {
      setError('Batch limit is 20 questions per request.')
      return
    }
    setLoading(true)
    setError(null)
    setResults([])
    try {
      const response = await axios.post('/api/ask/batch', { questions, threshold })
      setResults(response.data.results)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Batch request failed. Check the API and provider configuration.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Batch Ask</h1>
        <p className="page-desc">Submit up to 20 questions, one per line. Each result contains real responses or explicit provider errors.</p>
      </header>
      <section className="card">
        <form onSubmit={submit}>
          <label className="form-label" htmlFor="batch-questions">Questions</label>
          <textarea id="batch-questions" className="form-input question-input" rows={8} value={text}
            onChange={event => setText(event.target.value)} placeholder={'Question one?\nQuestion two?'} />
          <p className="form-hint" style={{ margin: '8px 0 16px' }}>Blank lines are ignored. Current threshold: {threshold.toFixed(2)}.</p>
          {error && <div className="alert alert-error" role="alert">{Array.isArray(error) ? error.map(item => item.msg).join(' ') : error}</div>}
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? 'Checking questions…' : 'Check batch'}
          </button>
        </form>
      </section>
      {loading && <div className="loading-container"><span className="loading-spinner" />Models are answering each question…</div>}
      {results.length > 0 && (
        <section className="card" style={{ marginTop: 20 }}>
          <div className="card-header"><h2 className="card-title">Batch results</h2><span className="badge badge-blue">{results.length} questions</span></div>
          <div className="table-wrapper">
            <table>
              <thead><tr><th>Question</th><th>Verdict</th><th>Consensus</th><th>Trust estimate</th><th>Provider errors</th></tr></thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={`${index}-${result.question}`}>
                    <td>{result.question}</td>
                    <td><span className={`badge badge-${result.verdict === 'TRUSTED' ? 'success' : result.verdict === 'DISPUTED' ? 'warning' : result.verdict === 'ABSTAIN' ? 'danger' : 'blue'}`}>{result.verdict || 'Not trained'}</span></td>
                    <td>{result.verdict === 'ABSTAIN' ? 'No model has a trustworthy answer.' : result.consensus_answer || '—'} ({result.answered_model_count}/{result.configured_model_count} answered)</td>
                    <td className="td-mono">{percent(result.trust_score)}</td>
                    <td>{result.models.filter(model => model.error).length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
