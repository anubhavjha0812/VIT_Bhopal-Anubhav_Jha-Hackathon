import { createContext, useContext } from 'react'

export type Theme = 'light' | 'dark'

export function initialTheme(): Theme {
  // index.html already set the class before the first paint; read it back.
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem('theme', theme)
  } catch {
    // private window or blocked storage: the theme still applies for this visit
  }
}

export const ThemeContext = createContext<Theme>('light')
export const useTheme = () => useContext(ThemeContext)

/** Axis and grid colours for ECharts, which does not read CSS classes. */
export function chartColors(theme: Theme) {
  return theme === 'dark'
    ? { text: '#cbd5e1', grid: '#1e293b', up: '#2dd4bf', down: '#fb7185', line: '#e2e8f0' }
    : { text: '#334155', grid: '#e2e8f0', up: '#0d9488', down: '#e11d48', line: '#0f172a' }
}
