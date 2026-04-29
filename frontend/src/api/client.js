const BASE = ''
let statusSocket = null
let reconnectTimer = null
let reconnectDelayMs = 500
const subscribers = new Set()
const statusListeners = new Set()
const errorListeners = new Set()

function notifyStatus(connected) {
  statusListeners.forEach((fn) => {
    try { fn(connected) } catch {}
  })
}

function connectStatusSocket() {
  if (statusSocket && (statusSocket.readyState === WebSocket.OPEN || statusSocket.readyState === WebSocket.CONNECTING)) {
    return
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  statusSocket = new WebSocket(`${protocol}://${host}/ws/status`)

  statusSocket.onopen = () => {
    reconnectDelayMs = 500
    notifyStatus(true)
  }

  statusSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      subscribers.forEach((fn) => {
        try { fn(data) } catch {}
      })
    } catch {
      // ignore malformed frames
    }
  }

  statusSocket.onerror = (err) => {
    errorListeners.forEach((fn) => {
      try { fn(err) } catch {}
    })
  }

  statusSocket.onclose = () => {
    notifyStatus(false)
    statusSocket = null
    if (subscribers.size === 0) return
    clearTimeout(reconnectTimer)
    const jitter = Math.floor(Math.random() * 250)
    reconnectTimer = setTimeout(connectStatusSocket, reconnectDelayMs + jitter)
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 5000)
  }
}

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

export async function sendAgenticChat(task, maxSteps = 10) {
  return apiFetch('/api/chat/agentic', {
    method: 'POST',
    body: JSON.stringify({ task, max_steps: maxSteps }),
  })
}

/**
 * Open a WebSocket to /ws/agent-progress and stream agentic task events.
 *
 * @param {string} task - The task description.
 * @param {number} maxSteps - Maximum number of steps (default 10).
 * @param {function} onEvent - Called with each parsed event object.
 * @param {function} onError - Called on error.
 * @returns {{ close: function }} Handle to close the socket early.
 */
export function createAgentProgressSocket(task, maxSteps = 10, onEvent, onError) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  const ws = new WebSocket(`${protocol}://${host}/ws/agent-progress`)

  ws.onopen = () => {
    ws.send(JSON.stringify({ task, max_steps: maxSteps }))
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (onEvent) onEvent(data)
    } catch {
      // ignore malformed frames
    }
  }

  ws.onerror = (err) => {
    if (onError) onError(err)
  }

  ws.onclose = () => {}

  return {
    close() { ws.close() },
  }
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
  if (onMessage) subscribers.add(onMessage)
  if (onError) errorListeners.add(onError)
  connectStatusSocket()

  return {
    close() {
      if (onMessage) subscribers.delete(onMessage)
      if (onError) errorListeners.delete(onError)
      if (subscribers.size > 0) return
      clearTimeout(reconnectTimer)
      reconnectTimer = null
      if (statusSocket) {
        statusSocket.close()
        statusSocket = null
      }
      notifyStatus(false)
    },
    onStatusChange(cb) {
      if (!cb) return () => {}
      statusListeners.add(cb)
      return () => statusListeners.delete(cb)
    },
  }
}
