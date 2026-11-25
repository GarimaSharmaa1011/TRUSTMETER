import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const SYSTEMS = [
  ['best_single_raw_confidence', 'Best single · raw confidence'],
  ['best_single_calibrated_confidence', 'Best single · calibrated'],
  ['majority_vote', 'Majority vote'],
  ['trustmeter', 'TrustMeter · calibrated + abstain'],
]

function percentage(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

export default function ModelComparison({ metrics }) {
  if (!metrics) return <div className="alert alert-info">Complete the ARC evaluation to compare test-set systems.</div>
  const rows = SYSTEMS.map(([key, name]) => ({ key, name, ...(metrics.systems?.[key] || {}) }))
  const chartData = ['accuracy', 'selective_accuracy', 'coverage'].map(metric => ({
    metric: metric.replaceAll('_', ' '),
    ...Object.fromEntries(rows.map(row => [row.name, row[metric] == null ? null : row[metric] * 100])),
  }))

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Model Comparison</h1>
        <p className="page-desc">
          All system metrics and curves use the held-out ARC test split only
          ({metrics.evaluation?.test_set_size ?? '—'} qualified questions). Abstained questions count as incorrect in overall accuracy.
        </p>
      </header>
      <section className="card" style={{ marginBottom: 20 }}>
        <div className="table-wrapper">
          <table>
            <thead><tr><th>System</th><th>Accuracy</th><th>Selective accuracy</th><th>Coverage</th><th>Abstention</th><th>ECE</th><th>Brier</th><th>AURC</th></tr></thead>
            <tbody>{rows.map(row => (
              <tr key={row.key}>
                <td>{row.name}</td>
                <td className="td-mono">{percentage(row.accuracy)}</td>
                <td className="td-mono">{percentage(row.selective_accuracy)}</td>
                <td className="td-mono">{percentage(row.coverage)}</td>
                <td className="td-mono">{percentage(row.abstention_rate)}</td>
                <td className="td-mono">{row.ece?.toFixed(4) ?? '—'}</td>
                <td className="td-mono">{row.brier_score?.toFixed(4) ?? '—'}</td>
                <td className="td-mono">{row.aurc?.toFixed(4) ?? '—'}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="form-hint" style={{ marginTop: 12 }}>
          Best single model selection uses train-split accuracy only. At the comparison threshold, single-model
          confidence and TrustMeter can abstain; majority vote always returns its plurality answer when responses exist.
        </p>
      </section>
      <section className="card">
        <div className="card-header"><h2 className="card-title">Test-set system metrics</h2></div>
        <div className="chart-container" style={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8eef5" />
              <XAxis dataKey="metric" angle={-15} textAnchor="end" interval={0} />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip formatter={value => value == null ? 'Not available' : `${Number(value).toFixed(1)}%`} />
              <Legend verticalAlign="top" />
              {rows.map((row, index) => (
                <Bar key={row.key} dataKey={row.name} fill={['#64748b', '#3b82f6', '#d97706', '#16a34a'][index]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}
