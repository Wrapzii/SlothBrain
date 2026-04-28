import { useState, useRef, useEffect, useCallback } from 'react'
import { sendChat } from '../api/client.js'

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  border: '#1e3a5f',
  userBubble: '#0f3460',
  assistantBubble: '#1e3a5f',
}

function Message({ msg }) {
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return

    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setInput('')
    setLoading(true)

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
          {['auto', 'watcher', 'main'].map((a) => (
            <button
              key={a}
              onClick={() => setAgent(a)}
              style={{
                background: agent === a ? COLORS.green : 'transparent',
                color: agent === a ? '#000' : COLORS.text,
                border: `1px solid ${COLORS.border}`,
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
            Start a conversation…
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {loading && <TypingIndicator />}
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
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
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
            background: loading || !input.trim() ? COLORS.accent : COLORS.green,
            color: '#000',
            border: 'none',
            borderRadius: 8,
            padding: '0 20px',
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            fontWeight: 700,
            fontSize: 14,
            opacity: loading || !input.trim() ? 0.6 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  )
}
