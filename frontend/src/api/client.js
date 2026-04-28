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
