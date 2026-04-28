import { useState, useEffect } from 'react'
import { getSettings, updateSettings } from '../api/client.js'

const COLORS = {
  bg: '#1a1a2e',
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  border: '#1e3a5f',
  inputBg: '#0d1b2a',
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
  })
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getSettings()
      .then((data) => setForm((prev) => ({ ...prev, ...data })))
      .catch(() => {})
  }, [])

  const handleChange = (e) => {
    const { name, value, type } = e.target
    setForm((prev) => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value,
    }))
  }

  const handleSave = async () => {
    setLoading(true)
    setStatus(null)
    try {
      await updateSettings(form)
      setStatus({ ok: true, msg: 'Settings saved successfully.' })
    } catch (err) {
      setStatus({ ok: false, msg: `Error: ${err.message}` })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 28,
      maxWidth: 640,
    }}>
      <div style={{ color: COLORS.text, fontSize: 18, fontWeight: 600, marginBottom: 24 }}>
        Configuration
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 24px' }}>
        <Field label="Llama Host" name="llama_host" value={form.llama_host} onChange={handleChange} />
        <Field label="Llama Port" name="llama_port" value={form.llama_port} onChange={handleChange} type="number" />
        <Field label="Watcher Slot" name="watcher_slot" value={form.watcher_slot} onChange={handleChange} type="number" />
        <Field label="Main Slot" name="main_slot" value={form.main_slot} onChange={handleChange} type="number" />
        <Field label="Watcher Context Size" name="watcher_context_size" value={form.watcher_context_size} onChange={handleChange} type="number" />
        <Field label="Main Context Size" name="main_context_size" value={form.main_context_size} onChange={handleChange} type="number" />
        <Field label="Idle KV Quant" name="idle_kv_quant" value={form.idle_kv_quant} onChange={handleChange} options={['q4', 'q6', 'q8']} />
        <Field label="Active KV Quant" name="active_kv_quant" value={form.active_kv_quant} onChange={handleChange} options={['q4', 'q6', 'q8']} />
        <Field label="VRAM Threshold (MB)" name="vram_threshold_mb" value={form.vram_threshold_mb} onChange={handleChange} type="number" />
        <Field label="Embedding Model" name="embedding_model" value={form.embedding_model} onChange={handleChange} />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 8 }}>
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
