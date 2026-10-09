import { useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { useApi } from './api'
import ThemeToggle from './components/ThemeToggle'
import Overview from './pages/Overview'
import { applyTheme, initialTheme, ThemeContext, type Theme } from './theme'
import type { Overview as OverviewData } from './types'

// A page gets a link here only once it exists (no empty sections).
const PAGES = [
  { path: '/', label: 'Overview' },
]

function Header({ theme, onTheme }: { theme: Theme; onTheme: (t: Theme) => void }) {
  const { data } = useApi<OverviewData>('/overview')
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 sm:px-6 dark:border-slate-800 dark:bg-slate-900">
      <div>
        <h1 className="text-lg font-semibold">AI Risk Engine</h1>
        {data && (
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Sentiment: <span className="font-mono">{data.engine_models.sentiment}</span> · Events:{' '}
            <span className="font-mono">{data.engine_models.events}</span> · Impact: {data.config.impact}
          </p>
        )}
      </div>
      <ThemeToggle theme={theme} onChange={onTheme} />
    </header>
  )
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const changeTheme = (t: Theme) => {
    applyTheme(t)
    setTheme(t)
  }

  return (
    <ThemeContext.Provider value={theme}>
      <BrowserRouter>
        <div className="min-h-screen md:flex">
          <nav className="border-b border-slate-200 bg-white px-4 py-3 md:w-52 md:shrink-0 md:border-r md:border-b-0 md:py-6 dark:border-slate-800 dark:bg-slate-900">
            <div className="mb-4 hidden text-xs font-semibold uppercase tracking-wider text-teal-700 md:block dark:text-teal-400">
              S&amp;P · Crisil 2026
            </div>
            <ul className="flex gap-2 overflow-x-auto whitespace-nowrap md:flex-col md:overflow-visible">
              {PAGES.map((p) => (
                <li key={p.path}>
                  <NavLink
                    to={p.path}
                    end
                    className={({ isActive }) =>
                      `block rounded-lg px-3 py-2 text-sm ${
                        isActive
                          ? 'bg-teal-50 font-semibold text-teal-800 dark:bg-teal-950/50 dark:text-teal-300'
                          : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'
                      }`
                    }
                  >
                    {p.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="min-w-0 flex-1">
            <Header theme={theme} onTheme={changeTheme} />
            <main className="p-4 sm:p-6">
              <Routes>
                <Route path="/" element={<Overview />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </ThemeContext.Provider>
  )
}
