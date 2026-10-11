import { useEffect, useState } from 'react'

const BASE = '/api'

export function getJSON<T>(path: string): Promise<T> {
  return request<T>(path)
}

export function postJSON<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + path, init)
  } catch {
    throw new Error('API not reachable. Start it with: uv run uvicorn riskengine.api:app --port 8000')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `API error ${res.status} on ${path}`)
  }
  return res.json() as Promise<T>
}

type State<T> = { data: T | null; error: string | null; loading: boolean }

/** Fetch on mount and whenever the path changes: { data, error, loading }. While a new path
 * loads, `data` keeps the last answer and `loading` is true. A null path waits. */
export function useApi<T>(path: string | null): State<T> {
  const [state, setState] = useState<{
    data: T | null
    error: string | null
    path: string | null
  }>({
    data: null,
    error: null,
    path: null,
  })
  useEffect(() => {
    if (path == null) return
    let live = true
    getJSON<T>(path).then(
      (data) => live && setState({ data, error: null, path }),
      (e: Error) => live && setState({ data: null, error: e.message, path }),
    )
    return () => {
      live = false
    }
  }, [path])
  return {
    data: state.data,
    error: state.error,
    loading: path != null && state.path !== path,
  }
}
