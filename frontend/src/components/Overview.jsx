function percent(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

export default function Overview({ metrics, providers = [], evaluationError }) {
  const dataset = metrics?.dataset_info
  const trust = metrics?.systems?.trustmeter
  const best = metrics?.best_single_model

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">TrustMeter — When Should an AI Admit It Is Unsure?</h1>
        <p className="page-desc">
          Compare configured model responses, estimate whether a plurality answer is correct,
          and abstain when the test-calibrated trust score is below your threshold.
        </p>
      </header>

      <div className="alert alert-warning">
        A live verdict is an estimate, not proof of correctness. Historical accuracy is measured on ARC-Challenge
        and may not transfer to other topics. Answer grouping uses normalized string matching.
      </div>

      {!metrics && (
        <div className="alert alert-info">
          {evaluationError || 'No evaluation results are saved yet.'} Provider responses and test metrics will appear only after
          the ARC evaluation completes successfully.
        </div>
      )}

      {metrics && (
        <>
          <section className="stats-grid" aria-label="ARC test metrics">
            <div className="stat-card">
              <div className="stat-label">Qualified ARC questions</div>
              <div className="stat-value">
                {metrics.evaluation?.qualified_questions ?? dataset?.sample_size ?? '—'}
                {dataset?.sample_size != null && ` / ${dataset.sample_size}`}
              </div>
              <div className="stat-sub">
                {metrics.evaluation?.qualified_split_sizes?.train ?? metrics.evaluation?.split_sizes?.train ?? '—'} train ·{' '}
                {metrics.evaluation?.qualified_split_sizes?.validation ?? metrics.evaluation?.split_sizes?.validation ?? '—'} validation ·{' '}
                {metrics.evaluation?.qualified_split_sizes?.test ?? metrics.evaluation?.split_sizes?.test ?? '—'} test used
              </div>
            </div>
            <div className="stat-card success">
              <div className="stat-label">TrustMeter test accuracy</div>
              <div className="stat-value">{percent(trust?.selective_accuracy)}</div>
              <div className="stat-sub">Answered test questions</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Test coverage</div>
              <div className="stat-value">{percent(trust?.coverage)}</div>
              <div className="stat-sub">At threshold {metrics.evaluation?.threshold?.toFixed(2) ?? '—'}</div>
            </div>
            <div className="stat-card warning">
              <div className="stat-label">Test abstention rate</div>
              <div className="stat-value">{percent(trust?.abstention_rate)}</div>
              <div className="stat-sub">At selected evaluation threshold</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Trust ECE / Brier</div>
              <div className="stat-value">{trust?.ece?.toFixed(3) ?? '—'} / {trust?.brier_score?.toFixed(3) ?? '—'}</div>
              <div className="stat-sub">Test-set calibration scores</div>
            </div>
          </section>

          <section className="card" style={{ marginBottom: 20 }}>
            <div className="card-header"><h2 className="card-title">Model accuracy on ARC train split</h2></div>
            <p className="form-hint">Train-split accuracy also selects the single-model baselines; displayed evaluation metrics below are held-out test results.</p>
            <div className="table-wrapper">
              <table>
                <thead><tr><th>Model</th><th>Train accuracy</th><th>Test accuracy</th><th>Test coverage</th></tr></thead>
                <tbody>
                  {Object.entries(metrics.per_model || {}).map(([id, model]) => (
                    <tr key={id}>
                      <td>{model.display_name}</td>
                      <td className="td-mono">{percent(model.train_accuracy)}</td>
                      <td className="td-mono">{percent(model.raw?.accuracy)}</td>
                      <td className="td-mono">{percent(model.raw?.coverage)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="form-hint" style={{ marginTop: 12 }}>
              Best single model selected using train data only: {best?.display_name ?? '—'}.
            </p>
          </section>
        </>
      )}

      <div className="grid-2">
        <section className="card">
          <div className="card-header"><h2 className="card-title">Research question and hypothesis</h2></div>
          <p>
            Can multiple-model agreement, self-reported confidence, and training-only historical accuracy
            better identify incorrect consensus answers than relying on one model&apos;s confidence?
          </p>
          <p style={{ marginTop: 12 }}>
            <strong>Hypothesis:</strong> combining multi-model agreement with calibrated confidence detects
            wrong answers better than any single model&apos;s confidence.
          </p>
          {metrics && (
            <p style={{ marginTop: 12 }}>
              On the completed test split, TrustMeter AURC was{' '}
              {metrics.systems?.trustmeter?.aurc?.toFixed(4) ?? '—'} and the best-single raw-confidence
              AURC was {metrics.systems?.best_single_raw_confidence?.aurc?.toFixed(4) ?? '—'}; its calibrated-confidence
              AURC was {metrics.systems?.best_single_calibrated_confidence?.aurc?.toFixed(4) ?? '—'}.
              {metrics.hypothesis_assessment?.supported == null
                ? ' The result is not sufficient to assess the stated hypothesis.'
                : metrics.hypothesis_assessment.supported
                  ? ' This split supports the hypothesis by the predeclared lower-AURC criterion.'
                  : ' This split does not support the hypothesis by the predeclared lower-AURC criterion.'}
            </p>
          )}
        </section>
        <section className="card">
          <div className="card-header"><h2 className="card-title">Dataset and configured providers</h2></div>
          <p>
            {dataset
              ? `${dataset.name} · ${dataset.source}`
              : 'ARC-Challenge test split · Hugging Face datasets · sample size and split details are recorded after evaluation.'}
          </p>
          <div className="provider-list">
            {providers.map(provider => (
              <div className="provider-row" key={provider.id}>
                <span>
                  {provider.display_name}
                  {provider.error && <small className="provider-status-error">{provider.error}</small>}
                </span>
                <span className={`badge ${provider.reachable === true ? 'badge-success' : provider.reachable === false ? 'badge-warning' : 'badge-blue'}`}>
                  {provider.reachable === true
                    ? 'Reachable'
                    : provider.reachable === false
                      ? provider.configured ? 'Unavailable' : 'Key missing'
                      : 'Not checked'}
                </span>
              </div>
            ))}
            {!providers.length && <p className="form-hint">Provider reachability appears when the API health check responds.</p>}
          </div>
          <p className="form-hint" style={{ marginTop: 12 }}>
            Local inference requires Ollama to be running with the configured model pulled. Hosted providers require their
            API keys in backend/.env.
          </p>
        </section>
      </div>
    </div>
  )
}
