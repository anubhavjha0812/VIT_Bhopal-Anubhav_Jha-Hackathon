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
    ? { text: '#b3bcc9', grid: '#222b40', up: '#4fa088', down: '#d06a59', line: '#e6eaef' }
    : { text: '#333d55', grid: '#d3d9e1', up: '#1f6f5c', down: '#b23a2e', line: '#172238' }
}
