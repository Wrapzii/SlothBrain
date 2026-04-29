import { useState, useEffect, useCallback, useRef, Suspense, lazy } from 'react'
import ApprovalQueue from './components/ApprovalQueue.jsx'
import { emergencyStop, createStatusSocket } from './api/client.js'

const Dashboard = lazy(() => import('./components/Dashboard.jsx'))
const ChatInterface = lazy(() => import('./components/ChatInterface.jsx'))
const Settings = lazy(() => import('./components/Settings.jsx'))
const BenchmarkSuite = lazy(() => import('./components/BenchmarkSuite.jsx'))
const AgentsTab = lazy(() => import('./components/AgentsTab.jsx'))

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  red: '#ff4d4d',
  border: '#1e3a5f',
}

const TABS = [
  { id: 'dashboard', label: '📊 Dashboard' },
  { id: 'chat', label: '💬 Chat' },
  { id: 'agents', label: '🤖 Agents' },
  { id: 'settings', label: '⚙️ Settings' },
  { id: 'benchmarks', label: '🧪 Benchmarks' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [pendingApprovals, setPendingApprovals] = useState(0)
  const [stopping, setStopping] = useState(false)
  const wsRef = useRef(null)

  const handleMessage = useCallback((data) => {
    if (typeof data.pending_approvals === 'number') {
      setPendingApprovals(data.pending_approvals)
    }
  }, [])

  useEffect(() => {
    wsRef.current = createStatusSocket(handleMessage, () => {})
    return () => wsRef.current?.close()
  }, [handleMessage])

  const handleEmergencyStop = async () => {
    if (!confirm('⚠️ EMERGENCY STOP: Kill llama-server and destroy all agents?')) return
    setStopping(true)
    try {
      const result = await emergencyStop()
      if (result?.pending_approval) {
        alert('Emergency stop queued for approval.')
      } else {
        alert('Emergency stop executed. All agents destroyed and server stopped.')
      }
    } catch (err) {
      alert(`Emergency stop failed: ${err.message}`)
    } finally {
      setStopping(false)
    }
  }

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
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: 1 }}>
            🧠 <span style={{ color: COLORS.green }}>Sloth</span>Brain
          </div>
          <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
            Local AI Assistant · llama.cpp · LanceDB Memory
          </div>
        </div>

        {/* Approval queue badge */}
        <ApprovalQueue pendingCount={pendingApprovals} />

        {/* Emergency stop */}
        <button
          onClick={handleEmergencyStop}
          disabled={stopping}
          style={{
            background: COLORS.red,
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '8px 18px',
            fontSize: 13,
            fontWeight: 700,
            cursor: stopping ? 'not-allowed' : 'pointer',
            opacity: stopping ? 0.7 : 1,
            letterSpacing: 0.5,
          }}
        >
          🛑 {stopping ? 'Stopping…' : 'Emergency Stop'}
        </button>
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
            {tab.id === 'agents' && pendingApprovals > 0 && (
              <span style={{
                background: COLORS.red,
                color: '#fff',
                borderRadius: '50%',
                width: 16,
                height: 16,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 10,
                fontWeight: 700,
                marginLeft: 6,
              }}>{pendingApprovals}</span>
            )}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main style={{ padding: '28px 32px', maxWidth: 1100, margin: '0 auto' }}>
        <Suspense fallback={<div style={{ color: '#888' }}>Loading…</div>}>
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'chat' && <ChatInterface />}
          {activeTab === 'agents' && <AgentsTab />}
          {activeTab === 'settings' && <Settings />}
          {activeTab === 'benchmarks' && <BenchmarkSuite />}
        </Suspense>
      </main>
    </div>
  )
}
