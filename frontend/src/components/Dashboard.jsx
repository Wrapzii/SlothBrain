import { useState, useEffect, useRef, useCallback } from 'react'
import { createStatusSocket, setMode } from '../api/client.js'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  red: '#ff4d4d',
  yellow: '#ffd700',
  border: '#1e3a5f',
}

function Card({ children, style }) {
  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 20,
      ...style,
    }}>
      {children}
    </div>
  )
}

function Badge({ label, color }) {
  return (
    <span style={{
      background: color,
      color: '#000',
      borderRadius: 12,
      padding: '3px 12px',
      fontWeight: 700,
      fontSize: 13,
      letterSpacing: 1,
    }}>
      {label.toUpperCase()}
    </span>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [cpuHistory, setCpuHistory] = useState([])
  const [connected, setConnected] = useState(false)
  const [modeChanging, setModeChanging] = useState(false)
  const wsRef = useRef(null)

  const handleMessage = useCallback((data) => {
    setStats(data)
    setConnected(true)
    setCpuHistory((prev) => {
      const next = [...prev, { t: new Date().toLocaleTimeString(), cpu: data.cpu_percent }]
      return next.length > 30 ? next.slice(-30) : next
    })
  }, [])

  useEffect(() => {
    wsRef.current = createStatusSocket(handleMessage, () => setConnected(false))
    return () => wsRef.current?.close()
  }, [handleMessage])

  const handleSetMode = async (mode) => {
    setModeChanging(true)
    try {
      await setMode(mode)
    } catch {
      // ignore
    } finally {
      setModeChanging(false)
    }
  }

  const ramPercent = stats
    ? Math.round((stats.ram_used_mb / stats.ram_total_mb) * 100)
    : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ color: COLORS.text, fontSize: 18, fontWeight: 600 }}>System Dashboard</span>
        <span style={{
          width: 10, height: 10, borderRadius: '50%',
          background: connected ? COLORS.green : COLORS.red,
          display: 'inline-block',
        }} />
        <span style={{ color: COLORS.text, fontSize: 12 }}>{connected ? 'Live' : 'Disconnected'}</span>
      </div>

      {/* Stats cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
        <Card>
          <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>CPU Usage</div>
          <div style={{ color: COLORS.green, fontSize: 32, fontWeight: 700 }}>
            {stats ? `${stats.cpu_percent}%` : '—'}
          </div>
        </Card>

        <Card>
          <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>RAM Usage</div>
          <div style={{ color: COLORS.text, fontSize: 24, fontWeight: 700 }}>
            {stats ? `${Math.round(stats.ram_used_mb)} / ${Math.round(stats.ram_total_mb)} MB` : '—'}
          </div>
          {stats && (
            <div style={{ marginTop: 8, background: COLORS.accent, borderRadius: 4, height: 8 }}>
              <div style={{
                background: ramPercent > 80 ? COLORS.red : COLORS.green,
                width: `${ramPercent}%`,
                height: '100%',
                borderRadius: 4,
                transition: 'width 0.5s ease',
              }} />
            </div>
          )}
          <div style={{ color: '#aaa', fontSize: 12, marginTop: 4 }}>{ramPercent}% used</div>
        </Card>

        <Card>
          <div style={{ color: '#aaa', fontSize: 12, marginBottom: 8 }}>Mode</div>
          <Badge
            label={stats?.mode ?? 'unknown'}
            color={stats?.mode === 'active' ? COLORS.green : COLORS.yellow}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <button
              onClick={() => handleSetMode('idle')}
              disabled={modeChanging || stats?.mode === 'idle'}
              style={{
                background: stats?.mode === 'idle' ? COLORS.accent : 'transparent',
                color: COLORS.text,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 6,
                padding: '5px 14px',
                cursor: 'pointer',
                fontSize: 13,
              }}
            >Idle</button>
            <button
              onClick={() => handleSetMode('active')}
              disabled={modeChanging || stats?.mode === 'active'}
              style={{
                background: stats?.mode === 'active' ? COLORS.green : 'transparent',
                color: stats?.mode === 'active' ? '#000' : COLORS.text,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 6,
                padding: '5px 14px',
                cursor: 'pointer',
                fontSize: 13,
              }}
            >Active</button>
          </div>
        </Card>
      </div>

      {/* CPU Chart */}
      <Card>
        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 12 }}>CPU % History (last 30 pts)</div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={cpuHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
            <XAxis dataKey="t" tick={{ fill: '#888', fontSize: 10 }} interval="preserveStartEnd" />
            <YAxis domain={[0, 100]} tick={{ fill: '#888', fontSize: 10 }} />
            <Tooltip
              contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
            />
            <Line
              type="monotone"
              dataKey="cpu"
              stroke={COLORS.green}
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Slot table */}
      <Card>
        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 12 }}>Slot Status</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', color: COLORS.text, fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${COLORS.border}` }}>
              <th style={{ textAlign: 'left', padding: '6px 12px', color: '#aaa' }}>Slot</th>
              <th style={{ textAlign: 'left', padding: '6px 12px', color: '#aaa' }}>Role</th>
              <th style={{ textAlign: 'left', padding: '6px 12px', color: '#aaa' }}>State</th>
            </tr>
          </thead>
          <tbody>
            {stats?.slots?.slots?.map((slot, idx) => (
              <tr key={idx} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                <td style={{ padding: '6px 12px' }}>{slot.id ?? idx}</td>
                <td style={{ padding: '6px 12px' }}>
                  {idx === stats?.slots?.watcher ? '👁 Watcher' : idx === stats?.slots?.main ? '🧠 Main' : '—'}
                </td>
                <td style={{ padding: '6px 12px' }}>
                  <Badge label={slot.state ?? 'unknown'} color={slot.state === 'processing' ? COLORS.green : '#555'} />
                </td>
              </tr>
            )) ?? (
              <tr>
                <td colSpan={3} style={{ padding: '12px', color: '#666', textAlign: 'center' }}>
                  No slot data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
