import { useState, useEffect, useCallback } from 'react'
import { listApprovals, approveAction, rejectAction } from '../api/client.js'

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

export default function ApprovalQueue({ pendingCount }) {
  const [open, setOpen] = useState(false)
  const [approvals, setApprovals] = useState([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(() => {
    listApprovals()
      .then((d) => setApprovals(d.approvals ?? []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (open) refresh()
  }, [open, refresh])

  // Auto-open when new approvals come in
  useEffect(() => {
    if (pendingCount > 0) refresh()
  }, [pendingCount, refresh])

  const handle = async (id, action) => {
    setLoading(true)
    try {
      if (action === 'approve') await approveAction(id)
      else await rejectAction(id)
      refresh()
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  const count = approvals.length

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: count > 0 ? COLORS.yellow : COLORS.accent,
          color: '#000',
          border: 'none',
          borderRadius: 6,
          padding: '6px 14px',
          fontWeight: 700,
          fontSize: 13,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        ✅ Approvals
        {count > 0 && (
          <span style={{
            background: COLORS.red,
            color: '#fff',
            borderRadius: '50%',
            width: 18,
            height: 18,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 11,
            fontWeight: 700,
          }}>{count}</span>
        )}
      </button>

      {open && (
        <div style={{
          position: 'absolute',
          top: '110%',
          right: 0,
          zIndex: 100,
          background: COLORS.card,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 8,
          minWidth: 340,
          maxWidth: 440,
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
          padding: 16,
        }}>
          <div style={{ color: COLORS.text, fontWeight: 700, marginBottom: 12 }}>
            Pending Approvals {count > 0 ? `(${count})` : ''}
          </div>
          {count === 0 ? (
            <div style={{ color: '#555', fontSize: 13 }}>No pending approvals.</div>
          ) : (
            approvals.map((a) => (
              <div key={a.id} style={{
                background: COLORS.inputBg,
                borderRadius: 6,
                padding: 12,
                marginBottom: 10,
                borderLeft: `3px solid ${COLORS.yellow}`,
              }}>
                <div style={{ color: COLORS.yellow, fontWeight: 600, fontSize: 13 }}>{a.action}</div>
                <div style={{ color: '#ccc', fontSize: 12, marginTop: 4, marginBottom: 8 }}>{a.description}</div>
                <div style={{ color: '#888', fontSize: 11, marginBottom: 8 }}>
                  {new Date(a.created_at).toLocaleString()}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    onClick={() => handle(a.id, 'approve')}
                    disabled={loading}
                    style={{ background: COLORS.green, color: '#000', border: 'none', borderRadius: 5, padding: '5px 12px', fontWeight: 700, cursor: 'pointer', fontSize: 12 }}
                  >Approve</button>
                  <button
                    onClick={() => handle(a.id, 'reject')}
                    disabled={loading}
                    style={{ background: COLORS.red, color: '#fff', border: 'none', borderRadius: 5, padding: '5px 12px', fontWeight: 700, cursor: 'pointer', fontSize: 12 }}
                  >Reject</button>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
