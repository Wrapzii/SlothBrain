import { useState, useEffect } from 'react'
import { getAuditLog } from '../api/client.js'

const COLORS = {
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  red: '#ff4d4d',
  yellow: '#ffd700',
  border: '#1e3a5f',
  inputBg: '#0d1b2a',
}

const ACTION_COLORS = {
  server_restart: COLORS.red,
  server_stop: COLORS.red,
  emergency_stop: '#ff0000',
  approval_granted: COLORS.green,
  approval_rejected: COLORS.red,
  preset_created: COLORS.green,
  preset_deleted: COLORS.red,
  agent_spawned: COLORS.green,
  agent_destroyed: COLORS.red,
}

export default function AuditLogViewer() {
  const [entries, setEntries] = useState([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [n, setN] = useState(50)

  const refresh = async () => {
    setLoading(true)
    try {
      const data = await getAuditLog(n)
      setEntries((data.entries ?? []).reverse())
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open) refresh()
  }, [open, n])

  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 20,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <span style={{ color: COLORS.text, fontWeight: 600, fontSize: 16 }}>📋 Audit Log</span>
        <button
          onClick={() => { setOpen(!open); }}
          style={{
            background: COLORS.accent,
            color: COLORS.text,
            border: 'none',
            borderRadius: 6,
            padding: '5px 14px',
            cursor: 'pointer',
            fontSize: 13,
          }}
        >{open ? 'Hide' : 'Show'}</button>
        {open && (
          <>
            <button
              onClick={refresh}
              disabled={loading}
              style={{
                background: 'transparent',
                color: COLORS.green,
                border: `1px solid ${COLORS.green}`,
                borderRadius: 6,
                padding: '4px 12px',
                cursor: 'pointer',
                fontSize: 12,
              }}
            >{loading ? 'Loading…' : '↻ Refresh'}</button>
            <select
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              style={{
                background: COLORS.inputBg,
                color: COLORS.text,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 5,
                padding: '4px 8px',
                fontSize: 12,
              }}
            >
              {[25, 50, 100, 200].map((v) => (
                <option key={v} value={v}>Last {v}</option>
              ))}
            </select>
          </>
        )}
      </div>

      {open && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, color: COLORS.text }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                <th style={{ textAlign: 'left', padding: '5px 10px', color: '#aaa' }}>Time</th>
                <th style={{ textAlign: 'left', padding: '5px 10px', color: '#aaa' }}>Action</th>
                <th style={{ textAlign: 'left', padding: '5px 10px', color: '#aaa' }}>Actor</th>
                <th style={{ textAlign: 'left', padding: '5px 10px', color: '#aaa' }}>Details</th>
              </tr>
            </thead>
            <tbody>
              {entries.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ padding: 16, color: '#555', textAlign: 'center' }}>
                    {loading ? 'Loading…' : 'No audit entries.'}
                  </td>
                </tr>
              ) : entries.map((e, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                  <td style={{ padding: '5px 10px', color: '#aaa', whiteSpace: 'nowrap' }}>
                    {new Date(e.timestamp).toLocaleString()}
                  </td>
                  <td style={{ padding: '5px 10px' }}>
                    <span style={{
                      color: ACTION_COLORS[e.action] ?? COLORS.text,
                      fontWeight: 600,
                    }}>{e.action}</span>
                  </td>
                  <td style={{ padding: '5px 10px', color: '#aaa' }}>{e.actor}</td>
                  <td style={{ padding: '5px 10px', color: '#ccc', maxWidth: 280, wordBreak: 'break-all' }}>
                    {e.details || (e.after ? JSON.stringify(e.after).slice(0, 80) : '—')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
