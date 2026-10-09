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

type State<T> = { data: T | null; error: string | null }

/** Fetch once on mount: { data, error }, both null while loading. */
export function useApi<T>(path: string): State<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null })
  useEffect(() => {
    let live = true
    getJSON<T>(path).then(
      (data) => live && setState({ data, error: null }),
      (e: Error) => live && setState({ data: null, error: e.message }),
    )
    return () => {
      live = false
    }
  }, [path])
  return state
}
