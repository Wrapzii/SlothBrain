const BASE = ''

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`HTTP ${res.status}: ${text}`)
  }
  // 204 No Content has no body
  if (res.status === 204) return null
  return res.json()
}

export async function getStatus() {
  return apiFetch('/api/status')
}

export async function getSettings() {
  return apiFetch('/api/settings')
}

export async function updateSettings(settings) {
  return apiFetch('/api/settings', {
    method: 'POST',
    body: JSON.stringify(settings),
  })
}

export async function sendChat(message, agent = 'auto') {
  return apiFetch('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, agent }),
  })
}

export async function setMode(mode) {
  return apiFetch('/api/mode', {
    method: 'POST',
    body: JSON.stringify({ mode }),
  })
}

export async function runBenchmark(type = 'all') {
  return apiFetch('/api/benchmark', {
    method: 'POST',
    body: JSON.stringify({ type }),
  })
}

export async function searchMemory(query) {
  return apiFetch(`/api/memory/search?q=${encodeURIComponent(query)}`)
}

// ---- Agent Presets ----
export async function listPresets() {
  return apiFetch('/api/presets')
}

export async function createPreset(preset) {
  return apiFetch('/api/presets', { method: 'POST', body: JSON.stringify(preset) })
}

export async function updatePreset(id, data) {
  return apiFetch(`/api/presets/${id}`, { method: 'PUT', body: JSON.stringify(data) })
}

export async function deletePreset(id) {
  return apiFetch(`/api/presets/${id}`, { method: 'DELETE' })
}

export async function spawnAgent(presetId) {
  return apiFetch(`/api/presets/${presetId}/spawn`, { method: 'POST' })
}

// ---- Running Agents ----
export async function listAgents() {
  return apiFetch('/api/agents')
}

export async function destroyAgent(agentId) {
  return apiFetch(`/api/agents/${agentId}`, { method: 'DELETE' })
}

export async function chatWithAgent(agentId, message) {
  return apiFetch(`/api/agents/${agentId}/chat`, {
    method: 'POST',
    body: JSON.stringify({ message }),
  })
}

// ---- Server Management ----
export async function getServerStatus() {
  return apiFetch('/api/server/status')
}

export async function restartServer() {
  return apiFetch('/api/server/restart', { method: 'POST' })
}

// ---- Approval Queue ----
export async function listApprovals() {
  return apiFetch('/api/approvals')
}

export async function approveAction(id) {
  return apiFetch(`/api/approvals/${id}/approve`, { method: 'POST' })
}

export async function rejectAction(id) {
  return apiFetch(`/api/approvals/${id}/reject`, { method: 'POST' })
}

// ---- Emergency Stop ----
export async function emergencyStop() {
  return apiFetch('/api/emergency-stop', { method: 'POST' })
}

// ---- Audit Log ----
export async function getAuditLog(n = 100) {
  return apiFetch(`/api/audit-log?n=${n}`)
}

export function createStatusSocket(onMessage, onError) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  const ws = new WebSocket(`${protocol}://${host}/ws/status`)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch {
      // ignore malformed frames
    }
  }

  ws.onerror = (err) => {
    if (onError) onError(err)
  }

  return ws
}
