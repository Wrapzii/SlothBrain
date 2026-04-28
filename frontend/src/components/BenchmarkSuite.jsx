import { useState } from 'react'
import { runBenchmark } from '../api/client.js'

const COLORS = {
  card: '#16213e',
  accent: '#0f3460',
  text: '#e0e0e0',
  green: '#00ff88',
  border: '#1e3a5f',
}

function Spinner() {
  return (
    <span style={{
      display: 'inline-block',
      width: 16,
      height: 16,
      border: `2px solid ${COLORS.border}`,
      borderTop: `2px solid ${COLORS.green}`,
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
      verticalAlign: 'middle',
      marginLeft: 8,
    }} />
  )
}

function ResultTable({ results }) {
  if (!results) return null

  if (Array.isArray(results)) {
    if (results.length === 0) return <div style={{ color: '#888' }}>No results.</div>
    const keys = Object.keys(results[0])
    return (
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, color: COLORS.text, marginTop: 8 }}>
        <thead>
          <tr>
            {keys.map((k) => (
              <th key={k} style={{ textAlign: 'left', padding: '5px 10px', color: '#aaa', borderBottom: `1px solid ${COLORS.border}` }}>
                {k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((row, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
              {keys.map((k) => (
                <td key={k} style={{ padding: '5px 10px' }}>
                  {String(row[k] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  // Object result
  return (
    <pre style={{
      background: '#0d1b2a',
      color: COLORS.green,
      padding: 12,
      borderRadius: 6,
      fontSize: 12,
      overflowX: 'auto',
      marginTop: 8,
    }}>
      {JSON.stringify(results, null, 2)}
    </pre>
  )
}

function BenchmarkSection({ title, type, onRun, result, running }) {
  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 20,
      marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <span style={{ color: COLORS.text, fontWeight: 600 }}>{title}</span>
        <button
          onClick={() => onRun(type)}
          disabled={running}
          style={{
            background: running ? COLORS.accent : COLORS.green,
            color: running ? COLORS.text : '#000',
            border: 'none',
            borderRadius: 6,
            padding: '7px 18px',
            cursor: running ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          {running ? <>Running{<Spinner />}</> : 'Run'}
        </button>
      </div>
      {result && <ResultTable results={result} />}
    </div>
  )
}

export default function BenchmarkSuite() {
  const [results, setResults] = useState({})
  const [running, setRunning] = useState({})

  const handleRun = async (type) => {
    setRunning((prev) => ({ ...prev, [type]: true }))
    try {
      const data = await runBenchmark(type)
      setResults((prev) => ({ ...prev, [type]: data.results }))
    } catch (err) {
      setResults((prev) => ({ ...prev, [type]: { error: err.message } }))
    } finally {
      setRunning((prev) => ({ ...prev, [type]: false }))
    }
  }

  return (
    <div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <div style={{ color: COLORS.text, fontSize: 18, fontWeight: 600, marginBottom: 20 }}>
        Benchmark Suite
      </div>
      <BenchmarkSection
        title="🚀 Inference Speed Test"
        type="speed"
        onRun={handleRun}
        result={results['speed']}
        running={running['speed']}
      />
      <BenchmarkSection
        title="💾 VRAM Benchmark"
        type="vram"
        onRun={handleRun}
        result={results['vram']}
        running={running['vram']}
      />
      <BenchmarkSection
        title="🔀 Slot Interference Test"
        type="slots"
        onRun={handleRun}
        result={results['slots']}
        running={running['slots']}
      />
      <BenchmarkSection
        title="🧪 Run All Benchmarks"
        type="all"
        onRun={handleRun}
        result={results['all']}
        running={running['all']}
      />
    </div>
  )
}
