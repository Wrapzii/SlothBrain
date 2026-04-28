import { useState } from 'react'
import Dashboard from './components/Dashboard.jsx'
import ChatInterface from './components/ChatInterface.jsx'
import Settings from './components/Settings.jsx'
import BenchmarkSuite from './components/BenchmarkSuite.jsx'

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  border: '#1e3a5f',
}

const TABS = [
  { id: 'dashboard', label: '📊 Dashboard' },
  { id: 'chat', label: '💬 Chat' },
  { id: 'settings', label: '⚙️ Settings' },
  { id: 'benchmarks', label: '🧪 Benchmarks' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  return (
    <div style={{
      minHeight: '100vh',
      background: COLORS.bg,
      color: COLORS.text,
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
    }}>
      {/* Header */}
      <header style={{
        background: COLORS.card,
        borderBottom: `1px solid ${COLORS.border}`,
        padding: '16px 32px',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
      }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: 1 }}>
            🧠 <span style={{ color: COLORS.green }}>Sloth</span>Brain
          </div>
          <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
            Local AI Assistant · llama.cpp · LanceDB Memory
          </div>
        </div>
      </header>

      {/* Nav tabs */}
      <nav style={{
        background: COLORS.card,
        borderBottom: `1px solid ${COLORS.border}`,
        display: 'flex',
        padding: '0 32px',
        gap: 4,
      }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              background: activeTab === tab.id ? COLORS.accent : 'transparent',
              color: activeTab === tab.id ? COLORS.green : '#888',
              border: 'none',
              borderBottom: activeTab === tab.id ? `2px solid ${COLORS.green}` : '2px solid transparent',
              padding: '12px 18px',
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: activeTab === tab.id ? 600 : 400,
              transition: 'color 0.2s, border-color 0.2s',
            }}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main style={{ padding: '28px 32px', maxWidth: 1100, margin: '0 auto' }}>
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'settings' && <Settings />}
        {activeTab === 'benchmarks' && <BenchmarkSuite />}
      </main>
    </div>
  )
}
