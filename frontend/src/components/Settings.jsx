import { useState, useEffect } from 'react'
import { getSettings, updateSettings, getServerStatus, restartServer } from '../api/client.js'

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

function SectionTitle({ children }) {
  return (
    <div style={{
      color: COLORS.green,
      fontSize: 13,
      fontWeight: 700,
      letterSpacing: 1,
      textTransform: 'uppercase',
      borderBottom: `1px solid ${COLORS.border}`,
      paddingBottom: 6,
      marginBottom: 14,
      marginTop: 24,
    }}>{children}</div>
  )
}

function Field({ label, name, value, onChange, type = 'text', options }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ color: '#aaa', fontSize: 12, display: 'block', marginBottom: 4 }}>
        {label}
      </label>
      {options ? (
        <select
          name={name}
          value={value}
          onChange={onChange}
          style={{
            background: COLORS.inputBg,
            color: COLORS.text,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6,
            padding: '8px 12px',
            fontSize: 14,
            width: '100%',
          }}
        >
          {options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          name={name}
          value={value}
          onChange={onChange}
          style={{
            background: COLORS.inputBg,
            color: COLORS.text,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6,
            padding: '8px 12px',
            fontSize: 14,
            width: '100%',
            boxSizing: 'border-box',
          }}
        />
      )}
    </div>
  )
}

function Toggle({ label, name, value, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
        <input
          type="checkbox"
          name={name}
          checked={!!value}
          onChange={onChange}
          style={{ width: 16, height: 16, accentColor: COLORS.green }}
        />
        <span style={{ color: COLORS.text, fontSize: 13 }}>{label}</span>
      </label>
    </div>
  )
}

export default function Settings() {
  const [form, setForm] = useState({
    llama_host: '127.0.0.1',
    llama_port: 8080,
    watcher_slot: 0,
    main_slot: 1,
    watcher_context_size: 4096,
    main_context_size: 32768,
    idle_kv_quant: 'q4',
    active_kv_quant: 'q8',
    vram_threshold_mb: 2048,
    embedding_model: 'all-MiniLM-L6-v2',
    llama_server_path: '',
    llama_server_args: [],
    max_restarts_per_hour: 3,
    max_context_size: 131072,
    max_slots: 8,
    require_approval_server_restart: true,
    require_approval_kv_cache_change: true,
    require_approval_large_context_increase: true,
  })
  const [serverArgsText, setServerArgsText] = useState('')
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState(null)
  const [restarting, setRestarting] = useState(false)
  const [restartMsg, setRestartMsg] = useState(null)

  useEffect(() => {
    getSettings()
      .then((data) => {
        setForm((prev) => ({ ...prev, ...data }))
        if (Array.isArray(data.llama_server_args)) {
          setServerArgsText(data.llama_server_args.join('\n'))
        }
      })
      .catch(() => {})
    getServerStatus()
      .then((d) => setServerStatus(d.status))
      .catch(() => {})
  }, [])

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    if (type === 'checkbox') {
      setForm((prev) => ({ ...prev, [name]: checked }))
    } else {
      setForm((prev) => ({
        ...prev,
        [name]: type === 'number' ? Number(value) : value,
      }))
    }
  }

  const handleSave = async () => {
    setLoading(true)
    setStatus(null)
    try {
      const payload = {
        ...form,
        llama_server_args: serverArgsText
          .split('\n')
          .map((s) => s.trim())
          .filter(Boolean),
      }
      const result = await updateSettings(payload)
      if (result?.pending_approval) {
        setStatus({ ok: true, msg: `Awaiting approval: ${result.pending_approval.action}` })
      } else {
        setStatus({ ok: true, msg: 'Settings saved successfully.' })
      }
    } catch (err) {
      setStatus({ ok: false, msg: `Error: ${err.message}` })
    } finally {
      setLoading(false)
    }
  }

  const handleRestartServer = async () => {
    setRestarting(true)
    setRestartMsg(null)
    try {
      const result = await restartServer()
      if (result?.pending_approval) {
        setRestartMsg({ ok: true, text: 'Restart queued for approval.' })
      } else {
        setRestartMsg({ ok: true, text: `Server status: ${result.status}` })
        setServerStatus(result.status)
      }
    } catch (err) {
      setRestartMsg({ ok: false, text: err.message })
    } finally {
      setRestarting(false)
    }
  }

  const serverStatusColor =
    serverStatus === 'running' ? COLORS.green
    : serverStatus === 'stopped' ? COLORS.red
    : COLORS.yellow

  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 28,
      maxWidth: 700,
    }}>
      <div style={{ color: COLORS.text, fontSize: 18, fontWeight: 600, marginBottom: 4 }}>
        Configuration
      </div>

      <SectionTitle>llama.cpp Connection</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 24px' }}>
        <Field label="Llama Host" name="llama_host" value={form.llama_host} onChange={handleChange} />
        <Field label="Llama Port" name="llama_port" value={form.llama_port} onChange={handleChange} type="number" />
        <Field label="Watcher Slot" name="watcher_slot" value={form.watcher_slot} onChange={handleChange} type="number" />
        <Field label="Main Slot" name="main_slot" value={form.main_slot} onChange={handleChange} type="number" />
        <Field label="Watcher Context Size" name="watcher_context_size" value={form.watcher_context_size} onChange={handleChange} type="number" />
        <Field label="Main Context Size" name="main_context_size" value={form.main_context_size} onChange={handleChange} type="number" />
        <Field label="Idle KV Quant" name="idle_kv_quant" value={form.idle_kv_quant} onChange={handleChange} options={['q4', 'q6', 'q8']} />
        <Field label="Active KV Quant" name="active_kv_quant" value={form.active_kv_quant} onChange={handleChange} options={['q4', 'q6', 'q8']} />
        <Field label="Embedding Model" name="embedding_model" value={form.embedding_model} onChange={handleChange} />
      </div>

      <SectionTitle>Server Management</SectionTitle>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <span style={{ color: '#aaa', fontSize: 13 }}>llama-server status:</span>
        <span style={{
          background: serverStatusColor,
          color: '#000',
          borderRadius: 12,
          padding: '2px 12px',
          fontWeight: 700,
          fontSize: 12,
        }}>{serverStatus ?? '—'}</span>
        <button
          onClick={handleRestartServer}
          disabled={restarting}
          style={{
            background: COLORS.yellow,
            color: '#000',
            border: 'none',
            borderRadius: 6,
            padding: '5px 14px',
            fontSize: 13,
            fontWeight: 700,
            cursor: restarting ? 'not-allowed' : 'pointer',
            opacity: restarting ? 0.7 : 1,
          }}
        >{restarting ? 'Restarting…' : '↺ Restart Server'}</button>
        {restartMsg && (
          <span style={{ fontSize: 12, color: restartMsg.ok ? COLORS.green : COLORS.red }}>
            {restartMsg.text}
          </span>
        )}
      </div>
      <Field
        label="llama-server Executable Path"
        name="llama_server_path"
        value={form.llama_server_path}
        onChange={handleChange}
      />
      <div style={{ marginBottom: 16 }}>
        <label style={{ color: '#aaa', fontSize: 12, display: 'block', marginBottom: 4 }}>
          Server Launch Args (one per line)
        </label>
        <textarea
          value={serverArgsText}
          onChange={(e) => setServerArgsText(e.target.value)}
          rows={4}
          placeholder={'-m /path/to/model.gguf\n--ctx-size 8192\n--n-gpu-layers 35'}
          style={{
            background: COLORS.inputBg,
            color: COLORS.text,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6,
            padding: '8px 12px',
            fontSize: 13,
            width: '100%',
            boxSizing: 'border-box',
            resize: 'vertical',
            fontFamily: 'monospace',
          }}
        />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 24px' }}>
        <Field label="Max Restarts / Hour" name="max_restarts_per_hour" value={form.max_restarts_per_hour} onChange={handleChange} type="number" />
      </div>

      <SectionTitle>Hard Resource Limits</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 24px' }}>
        <Field label="Max Context Size (tokens)" name="max_context_size" value={form.max_context_size} onChange={handleChange} type="number" />
        <Field label="Max Slots" name="max_slots" value={form.max_slots} onChange={handleChange} type="number" />
        <Field label="VRAM Threshold (MB)" name="vram_threshold_mb" value={form.vram_threshold_mb} onChange={handleChange} type="number" />
      </div>

      <SectionTitle>Require Human Approval For</SectionTitle>
      <Toggle label="Server restart" name="require_approval_server_restart" value={form.require_approval_server_restart} onChange={handleChange} />
      <Toggle label="KV cache quantisation change" name="require_approval_kv_cache_change" value={form.require_approval_kv_cache_change} onChange={handleChange} />
      <Toggle label="Large context size increase (>2×)" name="require_approval_large_context_increase" value={form.require_approval_large_context_increase} onChange={handleChange} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 20 }}>
        <button
          onClick={handleSave}
          disabled={loading}
          style={{
            background: COLORS.green,
            color: '#000',
            border: 'none',
            borderRadius: 6,
            padding: '10px 28px',
            fontSize: 14,
            fontWeight: 700,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? 'Saving…' : 'Save Settings'}
        </button>
        {status && (
          <span style={{ color: status.ok ? COLORS.green : '#ff4d4d', fontSize: 13 }}>
            {status.msg}
          </span>
        )}
      </div>
    </div>
  )
}
