import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

function percentage(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

export default function ReliabilityAnalysis({ metrics }) {
  if (!metrics) return <div className="alert alert-info">No reliability charts are available until a real ARC evaluation has been saved.</div>
  const trust = metrics.systems?.trustmeter
  const reliability = (metrics.reliability?.trustmeter || []).map(point => ({
    confidence: point.bin_center,
    observed: point.accuracy,
    perfect: point.bin_center,
  }))
  const histogram = metrics.confidence_distribution?.trustmeter
  const histogramData = (histogram?.counts || []).map((count, index) => ({
    confidence: histogram.bin_centers[index] * 100,
    count,
  }))
  const riskCoverage = (metrics.risk_coverage?.trustmeter || [])
    .filter(point => point.risk != null)
    .map(point => ({ coverage: point.coverage * 100, risk: point.risk * 100 }))

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Reliability Analysis</h1>
        <p className="page-desc">
          Reliability, confidence distribution, and risk/coverage plots use held-out test predictions
          ({metrics.evaluation?.test_set_size ?? '—'} qualified questions).
        </p>
      </header>
      <div className="alert alert-warning">Calibration on ARC-Challenge does not guarantee a live verdict is correct, especially on other topics.</div>
      <div className="metric-cards">
        <div className="metric-card"><div className="metric-card-value">{trust?.brier_score?.toFixed(4) ?? '—'}</div><div className="metric-card-label">TrustMeter Brier</div></div>
        <div className="metric-card"><div className="metric-card-value">{trust?.ece?.toFixed(4) ?? '—'}</div><div className="metric-card-label">TrustMeter ECE</div></div>
        <div className="metric-card"><div className="metric-card-value">{percentage(trust?.coverage)}</div><div className="metric-card-label">Coverage at evaluation threshold</div></div>
        <div className="metric-card"><div className="metric-card-value">{trust?.aurc?.toFixed(4) ?? '—'}</div><div className="metric-card-label">Area under risk-coverage</div></div>
      </div>
      <div className="grid-2">
        <section className="card">
          <div className="card-header"><h2 className="card-title">Trust reliability diagram</h2></div>
          <div className="chart-container" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={reliability} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8eef5" />
                <XAxis dataKey="confidence" type="number" domain={[0, 1]} tickFormatter={value => percentage(value)} />
                <YAxis type="number" domain={[0, 1]} tickFormatter={value => percentage(value)} />
                <Tooltip formatter={value => percentage(value)} />
                <Legend />
                <Line dataKey="perfect" name="Perfect calibration" stroke="#94a3b8" strokeDasharray="5 5" dot={false} />
                <Line dataKey="observed" name="Observed correctness" stroke="#2563eb" connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="form-hint">Each point is one confidence bin; counts come from the saved test evaluation.</p>
        </section>
        <section className="card">
          <div className="card-header"><h2 className="card-title">Trust score distribution</h2></div>
          <div className="chart-container" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8eef5" />
                <XAxis dataKey="confidence" tickFormatter={value => `${Math.round(value)}%`} />
                <YAxis />
                <Tooltip labelFormatter={value => `Score ${Number(value).toFixed(1)}%`} />
                <Bar dataKey="count" name="Test questions" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
      <section className="card" style={{ marginTop: 20 }}>
        <div className="card-header"><h2 className="card-title">Risk versus coverage</h2></div>
        <div className="chart-container" style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={riskCoverage}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8eef5" />
              <XAxis dataKey="coverage" type="number" domain={[0, 100]} unit="%" />
              <YAxis type="number" domain={[0, 100]} unit="%" />
              <Tooltip formatter={value => `${Number(value).toFixed(1)}%`} />
              <Line dataKey="risk" name="Error rate among answered" stroke="#dc2626" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}
