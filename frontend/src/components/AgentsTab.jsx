import { useState, useEffect } from 'react'
import {
  listPresets, createPreset, updatePreset, deletePreset, spawnAgent,
  listAgents, destroyAgent, chatWithAgent,
} from '../api/client.js'

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  red: '#ff4d4d',
  yellow: '#ffd700',
  border: '#1e3a5f',
  inputBg: '#0d1b2a',
}

const FIELD_STYLE = {
  background: COLORS.inputBg,
  color: COLORS.text,
  border: `1px solid ${COLORS.border}`,
  borderRadius: 6,
  padding: '7px 10px',
  fontSize: 13,
  width: '100%',
  boxSizing: 'border-box',
}

const BLANK_PRESET = {
  name: '',
  description: '',
  system_prompt: '',
  context_size: 8192,
  temperature: 0.7,
  max_tokens: 1024,
}

function PresetCard({ preset, onSaved, onDeleted, onSpawn }) {
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState(preset)
  const [saving, setSaving] = useState(false)
  const [spawning, setSpawning] = useState(false)
  const [msg, setMsg] = useState(null)

  const handleChange = (e) => {
    const { name, value, type } = e.target
    setForm((p) => ({ ...p, [name]: type === 'number' ? Number(value) : value }))
  }

  const handleSave = async () => {
    setSaving(true)
    setMsg(null)
    try {
      const updated = await updatePreset(preset.id, form)
      onSaved(updated)
      setEditing(false)
      setMsg({ ok: true, text: 'Saved.' })
    } catch (err) {
      setMsg({ ok: false, text: err.message })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!confirm(`Delete preset "${preset.name}"?`)) return
    await deletePreset(preset.id)
    onDeleted(preset.id)
  }

  const handleSpawn = async () => {
    setSpawning(true)
    setMsg(null)
    try {
      const agent = await spawnAgent(preset.id)
      onSpawn(agent)
      setMsg({ ok: true, text: `Agent spawned: ${agent.agent_id.slice(0, 8)}…` })
    } catch (err) {
      setMsg({ ok: false, text: err.message })
    } finally {
      setSpawning(false)
    }
  }

  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 16,
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
    }}>
      {editing ? (
        <>
          <input name="name" value={form.name} onChange={handleChange} placeholder="Name" style={FIELD_STYLE} />
          <input name="description" value={form.description} onChange={handleChange} placeholder="Description" style={FIELD_STYLE} />
          <textarea
            name="system_prompt"
            value={form.system_prompt}
            onChange={handleChange}
            placeholder="System prompt…"
            rows={4}
            style={{ ...FIELD_STYLE, resize: 'vertical' }}
          />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
            <label style={{ color: '#aaa', fontSize: 11 }}>
              Context Size
              <input type="number" name="context_size" value={form.context_size} onChange={handleChange} style={FIELD_STYLE} />
            </label>
            <label style={{ color: '#aaa', fontSize: 11 }}>
              Temperature
              <input type="number" name="temperature" value={form.temperature} step="0.1" onChange={handleChange} style={FIELD_STYLE} />
            </label>
            <label style={{ color: '#aaa', fontSize: 11 }}>
              Max Tokens
              <input type="number" name="max_tokens" value={form.max_tokens} onChange={handleChange} style={FIELD_STYLE} />
            </label>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Btn onClick={handleSave} disabled={saving} color={COLORS.green}>{saving ? 'Saving…' : 'Save'}</Btn>
            <Btn onClick={() => setEditing(false)} color="#555">Cancel</Btn>
          </div>
        </>
      ) : (
        <>
          <div style={{ color: COLORS.green, fontWeight: 700, fontSize: 15 }}>{preset.name}</div>
          {preset.description && <div style={{ color: '#aaa', fontSize: 12 }}>{preset.description}</div>}
          <div style={{
            background: COLORS.inputBg,
            borderRadius: 4,
            padding: '6px 8px',
            fontSize: 12,
            color: '#ccc',
            maxHeight: 80,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
          }}>{preset.system_prompt}</div>
          <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#aaa' }}>
            <span>ctx: {preset.context_size}</span>
            <span>temp: {preset.temperature}</span>
            <span>max_tok: {preset.max_tokens}</span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Btn onClick={() => setEditing(true)} color={COLORS.accent}>Edit</Btn>
            <Btn onClick={handleSpawn} disabled={spawning} color="#1a5a1a">{spawning ? 'Spawning…' : '▶ Spawn'}</Btn>
            <Btn onClick={handleDelete} color="#5a1a1a">Delete</Btn>
          </div>
        </>
      )}
      {msg && <div style={{ fontSize: 12, color: msg.ok ? COLORS.green : COLORS.red }}>{msg.text}</div>}
    </div>
  )
}

function NewPresetForm({ onCreated }) {
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState(BLANK_PRESET)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState(null)

  const handleChange = (e) => {
    const { name, value, type } = e.target
    setForm((p) => ({ ...p, [name]: type === 'number' ? Number(value) : value }))
  }

  const handleCreate = async () => {
    setSaving(true)
    setMsg(null)
    try {
      const preset = await createPreset(form)
      onCreated(preset)
      setForm(BLANK_PRESET)
      setOpen(false)
    } catch (err) {
      setMsg(err.message)
    } finally {
      setSaving(false)
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        style={{
          background: 'transparent',
          border: `2px dashed ${COLORS.border}`,
          borderRadius: 8,
          color: '#aaa',
          fontSize: 22,
          padding: '32px 16px',
          cursor: 'pointer',
          width: '100%',
        }}
      >＋ New Preset</button>
    )
  }

  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.green}`, borderRadius: 8, padding: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ color: COLORS.green, fontWeight: 700 }}>New Preset</div>
      <input name="name" value={form.name} onChange={handleChange} placeholder="Name *" style={FIELD_STYLE} />
      <input name="description" value={form.description} onChange={handleChange} placeholder="Description" style={FIELD_STYLE} />
      <textarea
        name="system_prompt"
        value={form.system_prompt}
        onChange={handleChange}
        placeholder="System prompt *"
        rows={4}
        style={{ ...FIELD_STYLE, resize: 'vertical' }}
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        <label style={{ color: '#aaa', fontSize: 11 }}>Context Size<input type="number" name="context_size" value={form.context_size} onChange={handleChange} style={FIELD_STYLE} /></label>
        <label style={{ color: '#aaa', fontSize: 11 }}>Temperature<input type="number" name="temperature" value={form.temperature} step="0.1" onChange={handleChange} style={FIELD_STYLE} /></label>
        <label style={{ color: '#aaa', fontSize: 11 }}>Max Tokens<input type="number" name="max_tokens" value={form.max_tokens} onChange={handleChange} style={FIELD_STYLE} /></label>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <Btn onClick={handleCreate} disabled={saving} color={COLORS.green}>{saving ? 'Creating…' : 'Create'}</Btn>
        <Btn onClick={() => setOpen(false)} color="#555">Cancel</Btn>
      </div>
      {msg && <div style={{ color: COLORS.red, fontSize: 12 }}>{msg}</div>}
    </div>
  )
}

function RunningAgentRow({ agent, onDestroyed }) {
  const [chatOpen, setChatOpen] = useState(false)
  const [message, setMessage] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  const handleChat = async () => {
    if (!message.trim()) return
    setLoading(true)
    try {
      const res = await chatWithAgent(agent.agent_id, message)
      setResponse(res.response)
    } catch (err) {
      setResponse(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleDestroy = async () => {
    if (!confirm(`Kill agent "${agent.name}"?`)) return
    await destroyAgent(agent.agent_id)
    onDestroyed(agent.agent_id)
  }

  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 14, marginBottom: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ color: COLORS.green, fontWeight: 700 }}>{agent.name}</span>
        <span style={{ color: '#aaa', fontSize: 12 }}>{agent.agent_id.slice(0, 8)}…</span>
        <span style={{ color: '#aaa', fontSize: 12 }}>ctx:{agent.context_size} temp:{agent.temperature}</span>
        <Btn onClick={() => setChatOpen(!chatOpen)} color={COLORS.accent}>💬 Chat</Btn>
        <Btn onClick={handleDestroy} color="#5a1a1a">🗑 Kill</Btn>
      </div>
      {chatOpen && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleChat()}
              placeholder="Message this agent…"
              style={{ ...FIELD_STYLE, flex: 1 }}
            />
            <Btn onClick={handleChat} disabled={loading} color={COLORS.green}>{loading ? '…' : 'Send'}</Btn>
          </div>
          {response && (
            <div style={{ background: COLORS.inputBg, borderRadius: 4, padding: '8px 10px', fontSize: 13, color: COLORS.text, whiteSpace: 'pre-wrap' }}>
              {response}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Btn({ onClick, disabled, color, children }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: color,
        color: color === COLORS.green ? '#000' : COLORS.text,
        border: 'none',
        borderRadius: 6,
        padding: '6px 14px',
        fontSize: 13,
        fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.7 : 1,
      }}
    >
      {children}
    </button>
  )
}

export default function AgentsTab() {
  const [presets, setPresets] = useState([])
  const [agents, setAgents] = useState([])
  const [loadingPresets, setLoadingPresets] = useState(true)
  const [loadingAgents, setLoadingAgents] = useState(true)

  useEffect(() => {
    listPresets().then((d) => setPresets(d.presets ?? [])).catch(() => {}).finally(() => setLoadingPresets(false))
    listAgents().then((d) => setAgents(d.agents ?? [])).catch(() => {}).finally(() => setLoadingAgents(false))
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Presets Panel */}
      <section>
        <div style={{ color: COLORS.text, fontSize: 17, fontWeight: 600, marginBottom: 14 }}>🗂 Agent Presets</div>
        {loadingPresets ? (
          <div style={{ color: '#aaa' }}>Loading presets…</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
            {presets.map((p) => (
              <PresetCard
                key={p.id}
                preset={p}
                onSaved={(updated) => setPresets((ps) => ps.map((x) => x.id === updated.id ? updated : x))}
                onDeleted={(id) => setPresets((ps) => ps.filter((x) => x.id !== id))}
                onSpawn={(agent) => setAgents((a) => [...a, agent])}
              />
            ))}
            <NewPresetForm onCreated={(p) => setPresets((ps) => [...ps, p])} />
          </div>
        )}
      </section>

      {/* Running Agents Panel */}
      <section>
        <div style={{ color: COLORS.text, fontSize: 17, fontWeight: 600, marginBottom: 14 }}>
          🤖 Running Agents
          <span style={{ background: COLORS.accent, borderRadius: 12, padding: '2px 10px', fontSize: 13, marginLeft: 10 }}>
            {agents.length}
          </span>
        </div>
        {loadingAgents ? (
          <div style={{ color: '#aaa' }}>Loading agents…</div>
        ) : agents.length === 0 ? (
          <div style={{ color: '#555', fontSize: 14 }}>No agents running. Spawn one from a preset above.</div>
        ) : (
          agents.map((a) => (
            <RunningAgentRow
              key={a.agent_id}
              agent={a}
              onDestroyed={(id) => setAgents((as) => as.filter((x) => x.agent_id !== id))}
            />
          ))
        )}
      </section>
    </div>
  )
}
