import { useState, useRef, useEffect, useCallback } from 'react'
import { sendChat, createAgentProgressSocket } from '../api/client.js'

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  border: '#1e3a5f',
  userBubble: '#0f3460',
  assistantBubble: '#1e3a5f',
  agenticAccent: '#7b2fff',
}

const STEP_STATUS_ICONS = {
  pending: '○',
  running: '◌',
  complete: '●',
  failed: '✗',
}

const AGENTIC_STATUS_LABELS = {
  starting:  '⏳ Starting…',
  planning:  '🤔 Planning task…',
  running:   '⚡ Executing steps…',
  verifying: '🔍 Verifying completion…',
  complete:  '✅ Task complete',
  aborted:   '⛔ Task aborted',
  escalated: '🚨 Escalated to user',
  error:     '❌ Error',
}

const AGENTIC_STATUS_COLORS = {
  starting:  '#aaa',
  planning:  '#ffd700',
  running:   '#00aaff',
  verifying: '#ffa500',
  complete:  COLORS.green,
  aborted:   '#ff4444',
  escalated: '#ff8800',
  error:     '#ff4444',
}

// ---------------------------------------------------------------------------
// Helper: apply a progress event to the in-flight agentic message
// ---------------------------------------------------------------------------
function applyAgenticEvent(msg, event) {
  switch (event.type) {
    case 'planning':
      return { ...msg, status: 'planning' }

    case 'plan_ready':
      return {
        ...msg,
        status: 'running',
        approach: event.approach || '',
        steps: (event.steps || []).map((desc, i) => ({
          step_num: i + 1,
          description: desc,
          status: 'pending',
          result: '',
          watcher_feedback: '',
          screenshots: [],
          retries: 0,
        })),
      }

    case 'step_start':
      return {
        ...msg,
        steps: msg.steps.map((s) =>
          s.step_num === event.step_num ? { ...s, status: 'running' } : s,
        ),
      }

    case 'step_monitored':
      return {
        ...msg,
        steps: msg.steps.map((s) =>
          s.step_num === event.step_num
            ? { ...s, watcher_feedback: event.feedback || '' }
            : s,
        ),
      }

    case 'step_retry':
      return {
        ...msg,
        steps: msg.steps.map((s) =>
          s.step_num === event.step_num ? { ...s, retries: (s.retries || 0) + 1 } : s,
        ),
      }

    case 'step_complete': {
      const updated = {
        ...msg,
        steps: msg.steps.map((s) =>
          s.step_num === event.step_num ? { ...s, ...event } : s,
        ),
      }
      return updated
    }

    case 'verifying':
      return { ...msg, status: 'verifying' }

    case 'complete':
      return {
        ...msg,
        status: 'complete',
        summary: event.summary || '',
        verified: !!event.verified,
      }

    case 'aborted':
      return { ...msg, status: 'aborted', summary: event.reason || '' }

    case 'escalated':
      return { ...msg, status: 'escalated', summary: event.reason || '' }

    case 'supervisor_intervention':
      return {
        ...msg,
        supervisorEvents: [
          ...(msg.supervisorEvents || []),
          { step_num: event.step_num, action: event.action, message: event.message },
        ],
      }

    case 'context_reset':
      return {
        ...msg,
        supervisorEvents: [
          ...(msg.supervisorEvents || []),
          { step_num: null, action: 'context_reset', message: `Restored to step ${event.restored_to_step}: ${event.message}` },
        ],
        // Mark all steps at or after restored_to_step as pending again
        steps: msg.steps.map((s) =>
          s.step_num >= event.restored_to_step ? { ...s, status: 'pending', result: '' } : s,
        ),
      }

    case 'result':
      return {
        ...msg,
        status: msg.status === 'running' || msg.status === 'verifying' ? 'complete' : msg.status,
        summary: event.summary || msg.summary,
        verified: event.completion_verified ?? msg.verified,
        steps: event.steps || msg.steps,
      }

    case 'error':
      return { ...msg, status: 'error', summary: event.message || 'Unknown error' }

    default:
      return msg
  }
}

// ---------------------------------------------------------------------------
// AgenticMessage component
// ---------------------------------------------------------------------------
function AgenticMessage({ msg }) {
  const statusColor = AGENTIC_STATUS_COLORS[msg.status] || '#aaa'
  const statusLabel = AGENTIC_STATUS_LABELS[msg.status] || msg.status

  return (
    <div style={{
      background: COLORS.assistantBubble,
      color: COLORS.text,
      borderRadius: '16px 16px 16px 4px',
      padding: '12px 16px',
      maxWidth: '85%',
      fontSize: 14,
      lineHeight: 1.5,
    }}>
      {/* Header */}
      <div style={{ color: COLORS.green, fontSize: 11, marginBottom: 6, fontWeight: 700 }}>
        [AGENTIC]
      </div>

      {/* Approach */}
      {msg.approach && (
        <div style={{ color: '#bbb', fontSize: 12, marginBottom: 8, fontStyle: 'italic' }}>
          {msg.approach}
        </div>
      )}

      {/* Steps */}
      {msg.steps && msg.steps.length > 0 && (
        <div style={{
          borderTop: `1px solid ${COLORS.border}`,
          paddingTop: 8,
          marginTop: 4,
          marginBottom: 8,
        }}>
          {msg.steps.map((step) => (
            <div key={step.step_num} style={{ marginBottom: 10 }}>
              {/* Step header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{
                  color: step.status === 'complete' ? COLORS.green
                       : step.status === 'running'  ? '#00aaff'
                       : step.status === 'failed'   ? '#ff4444'
                       : '#666',
                  fontSize: 13,
                  fontWeight: 700,
                }}>
                  {STEP_STATUS_ICONS[step.status] || '○'}
                </span>
                <span style={{
                  color: COLORS.text,
                  fontSize: 13,
                  fontWeight: step.status === 'running' ? 700 : 400,
                }}>
                  {step.step_num}. {step.description}
                  {step.retries > 0 && (
                    <span style={{ color: '#ffd700', fontSize: 11 }}> (retry {step.retries})</span>
                  )}
                </span>
              </div>

              {/* Step result preview */}
              {step.result && step.status !== 'pending' && (
                <div style={{
                  color: '#bbb',
                  fontSize: 12,
                  marginTop: 3,
                  marginLeft: 20,
                  maxHeight: 72,
                  overflow: 'hidden',
                }}>
                  {step.result.slice(0, 200)}{step.result.length > 200 ? '…' : ''}
                </div>
              )}

              {/* Watcher feedback */}
              {step.watcher_feedback && (
                <div style={{
                  color: '#ffd700',
                  fontSize: 11,
                  marginTop: 2,
                  marginLeft: 20,
                  fontStyle: 'italic',
                }}>
                  Watcher: {step.watcher_feedback.slice(0, 120)}{step.watcher_feedback.length > 120 ? '…' : ''}
                </div>
              )}

              {/* Screenshots */}
              {step.screenshots && step.screenshots.length > 0 && (
                <div style={{ marginTop: 6, marginLeft: 20 }}>
                  {step.screenshots.map((b64, i) => (
                    <img
                      key={i}
                      src={`data:image/png;base64,${b64}`}
                      alt={`Step ${step.step_num} screenshot`}
                      style={{ maxWidth: '100%', borderRadius: 4, marginTop: 4, display: 'block' }}
                    />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Status badge */}
      <div style={{ color: statusColor, fontSize: 12, fontWeight: 600 }}>
        {statusLabel}
      </div>

      {/* Summary */}
      {msg.summary && (
        <div style={{ color: COLORS.text, fontSize: 13, marginTop: 6, whiteSpace: 'pre-wrap' }}>
          {msg.summary}
        </div>
      )}

      {/* Supervisor events */}
      {msg.supervisorEvents && msg.supervisorEvents.length > 0 && (
        <div style={{
          borderTop: `1px solid ${COLORS.border}`,
          paddingTop: 8,
          marginTop: 8,
        }}>
          <div style={{ color: '#ff8800', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
            🛡 Safety Supervisor
          </div>
          {msg.supervisorEvents.map((ev, i) => (
            <div key={i} style={{ color: '#ff8800', fontSize: 11, marginBottom: 2 }}>
              [{ev.action}]{ev.step_num !== null && ev.step_num !== undefined ? ` step ${ev.step_num}` : ''}: {(ev.message || '').slice(0, 120)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Regular message component
// ---------------------------------------------------------------------------
function Message({ msg }) {
  if (msg.type === 'agentic') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
        <AgenticMessage msg={msg} />
      </div>
    )
  }

  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12,
    }}>
      <div style={{
        background: isUser ? COLORS.userBubble : COLORS.assistantBubble,
        color: COLORS.text,
        borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
        padding: '10px 16px',
        maxWidth: '70%',
        fontSize: 14,
        lineHeight: 1.5,
      }}>
        {!isUser && msg.agent && (
          <div style={{ color: COLORS.green, fontSize: 11, marginBottom: 4, fontWeight: 700 }}>
            [{msg.agent.toUpperCase()}]
          </div>
        )}
        <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div style={{ display: 'flex', gap: 4, padding: '10px 16px', alignItems: 'center' }}>
      {[0, 1, 2].map((i) => (
        <span key={i} style={{
          width: 8, height: 8, borderRadius: '50%',
          background: COLORS.green,
          animation: `bounce 1s ease-in-out ${i * 0.2}s infinite`,
          display: 'inline-block',
        }} />
      ))}
    </div>
  )
}

export default function ChatInterface() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [agent, setAgent] = useState('auto')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const wsRef = useRef(null)
  const agenticIdRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return

    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setInput('')
    setLoading(true)

    if (agent === 'agentic') {
      // ── Agentic mode: stream progress via WebSocket ──────────────────────
      const msgId = Date.now()
      agenticIdRef.current = msgId
      setMessages((prev) => [
        ...prev,
        {
          id: msgId,
          role: 'assistant',
          type: 'agentic',
          task: text,
          status: 'starting',
          approach: '',
          steps: [],
          summary: '',
          verified: false,
        },
      ])

      const socket = createAgentProgressSocket(
        text,
        10,
        (event) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === msgId ? applyAgenticEvent(m, event) : m,
            ),
          )
          // Close WebSocket once we receive the final result
          if (event.type === 'result' || event.type === 'error') {
            socket.close()
            setLoading(false)
          }
        },
        () => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === msgId
                ? { ...m, status: 'error', summary: 'WebSocket connection failed' }
                : m,
            ),
          )
          setLoading(false)
        },
      )
      wsRef.current = socket
      return
    }

    // ── Standard mode ────────────────────────────────────────────────────
    try {
      const data = await sendChat(text, agent)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          agent: data.agent,
          handoff: data.handoff,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}`, agent: 'system' },
      ])
    } finally {
      setLoading(false)
    }
  }, [input, loading, agent])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const allAgents = ['auto', 'watcher', 'main', 'agentic']

  /**
   * Return the active background colour for an agent mode button.
   */
  function agentButtonBg(a) {
    if (agent !== a) return 'transparent'
    return a === 'agentic' ? COLORS.agenticAccent : COLORS.green
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '70vh',
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
      `}</style>

      {/* Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '12px 16px',
        borderBottom: `1px solid ${COLORS.border}`,
        gap: 12,
      }}>
        <span style={{ color: COLORS.text, fontWeight: 600, fontSize: 15 }}>Chat</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: '#aaa', fontSize: 12 }}>Agent:</span>
          {allAgents.map((a) => (
            <button
              key={a}
              onClick={() => setAgent(a)}
              title={a === 'agentic' ? 'Multi-step agentic loop with watcher monitoring' : undefined}
              style={{
                background: agentButtonBg(a),
                color: agent === a ? '#fff' : COLORS.text,
                border: `1px solid ${a === 'agentic' ? COLORS.agenticAccent : COLORS.border}`,
                borderRadius: 6,
                padding: '4px 12px',
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: agent === a ? 700 : 400,
              }}
            >
              {a}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        {messages.length === 0 && (
          <div style={{ color: '#555', textAlign: 'center', marginTop: 40, fontSize: 14 }}>
            {agent === 'agentic'
              ? 'Describe a task and the agent will plan and execute it step-by-step…'
              : 'Start a conversation…'}
          </div>
        )}
        {messages.map((msg, i) => <Message key={msg.id || i} msg={msg} />)}
        {loading && agent !== 'agentic' && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        display: 'flex',
        padding: '12px 16px',
        borderTop: `1px solid ${COLORS.border}`,
        gap: 10,
      }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            agent === 'agentic'
              ? 'Describe a task for the agentic loop… (Enter to start)'
              : 'Type a message… (Enter to send, Shift+Enter for newline)'
          }
          rows={2}
          style={{
            flex: 1,
            background: '#0d1b2a',
            color: COLORS.text,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            padding: '10px 14px',
            fontSize: 14,
            resize: 'none',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            background: loading || !input.trim()
              ? COLORS.accent
              : agentButtonBg(agent) || COLORS.green,
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            padding: '0 20px',
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            fontWeight: 700,
            fontSize: 14,
            opacity: loading || !input.trim() ? 0.6 : 1,
          }}
        >
          {agent === 'agentic' ? 'Run' : 'Send'}
        </button>
      </div>
    </div>
  )
}
