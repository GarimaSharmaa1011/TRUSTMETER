const TABS = [
  { id: 'overview', label: 'Overview', icon: '🏠' },
  { id: 'ask', label: 'Ask a Question', icon: '💬' },
  { id: 'batch', label: 'Batch Ask', icon: '☷' },
  { id: 'leaderboard', label: 'Leaderboard', icon: '🏆' },
  { id: 'comparison', label: 'Comparison', icon: '📊' },
  { id: 'reliability', label: 'Reliability Analysis', icon: '📈' },
  { id: 'threshold', label: 'Threshold Settings', icon: '⚙️' },
]

export default function Navbar({ activeTab, onTabChange, backendStatus }) {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <div className="brand-icon">T</div>
          <div>
            <div className="brand-text">TrustMeter</div>
            <div className="brand-subtitle">AI Uncertainty Research</div>
          </div>
        </div>

        <div className="navbar-tabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab${activeTab === tab.id ? ' active' : ''}`}
              onClick={() => onTabChange(tab.id)}
              type="button"
            >
              <span className="tab-icon">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        <div className={`navbar-status ${backendStatus ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {backendStatus ? 'API Online' : 'API Offline'}
        </div>
      </div>
    </nav>
  )
}
