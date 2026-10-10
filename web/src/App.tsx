import { useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { useApi } from './api'
import ThemeToggle from './components/ThemeToggle'
import Analyze from './pages/Analyze'
import Engine from './pages/Engine'
import ModuleA from './pages/ModuleA'
import ModuleB from './pages/ModuleB'
import Overview from './pages/Overview'
import Replay from './pages/Replay'
import { applyTheme, initialTheme, ThemeContext, type Theme } from './theme'
import type { Overview as OverviewData } from './types'

// A page gets a link here only once it exists (no empty sections).
const PAGES = [
  { path: '/', label: 'Overview' },
  { path: '/engine', label: 'Engine' },
  { path: '/analyze', label: 'Analyze' },
  { path: '/replay', label: 'Live replay' },
  { path: '/module-a', label: 'Module A' },
  { path: '/module-b', label: 'Module B' },
]

/** Floating page switcher: a small pill with the current page that opens into every page and the
 *  models the engine runs. Top centre on wide screens, bottom on phones. */
function Island() {
  const { data } = useApi<OverviewData>('/overview')
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  const current = PAGES.find((p) => p.path === pathname) ?? PAGES[0]

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-[calc(env(safe-area-inset-bottom,0px)+12px)] z-50 flex justify-center px-3 md:top-[calc(env(safe-area-inset-top,0px)+12px)] md:bottom-auto">
      <nav
        aria-label="Pages"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setOpen(false)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
        style={{ maxWidth: open ? '60rem' : '19rem' }}
        className="island pointer-events-auto w-max overflow-hidden rounded-[28px] border border-white/10 bg-slate-950/90 text-white shadow-xl shadow-slate-900/25 backdrop-blur-xl transition-[max-width,box-shadow] duration-[750ms] ease-[cubic-bezier(0.32,0.72,0,1)] hover:shadow-2xl dark:bg-black/85 dark:shadow-black/50"
      >
        {open ? (
          <div className="animate-[island-in_550ms_cubic-bezier(0.32,0.72,0,1)_120ms_both] p-1.5">
            <div className="flex items-center gap-1">
              <ul className="flex min-w-0 gap-1 overflow-x-auto whitespace-nowrap">
                {PAGES.map((p) => (
                  <li key={p.path}>
                    <NavLink
                      to={p.path}
                      end
                      onClick={() => setOpen(false)}
                      className={({ isActive }) =>
                        `block rounded-full px-3.5 py-1.5 text-sm transition ${
                          isActive ? 'bg-white font-semibold text-slate-900' : 'text-white/75 hover:bg-white/10 hover:text-white'
                        }`
                      }
                    >
                      {p.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
            {data && (
              <p className="truncate px-3.5 pt-1 pb-0.5 text-[11px] text-white/50">
                AI Risk Engine · S&amp;P · Crisil 2026 · sentiment <span className="font-mono">{data.engine_models.sentiment}</span>{' '}
                · events <span className="font-mono">{data.engine_models.events}</span> · impact {data.config.impact}
              </p>
            )}
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setOpen(true)}
            style={{ animation: 'island-in 450ms cubic-bezier(0.32,0.72,0,1) both' }}
            aria-expanded={false}
            className="flex w-full items-center gap-2.5 py-2 pr-4 pl-3 text-sm whitespace-nowrap"
          >
            <span className="h-2 w-2 shrink-0 rounded-full bg-teal-400 shadow-[0_0_8px] shadow-teal-400" />
            <span className="text-white/60">AI Risk Engine</span>
            <span className="font-semibold">{current.label}</span>
            <svg viewBox="0 0 20 20" className="h-4 w-4 text-white/50" fill="currentColor" aria-hidden>
              <path d="M5.2 7.7a.75.75 0 0 1 1.06.04L10 11.7l3.74-3.96a.75.75 0 1 1 1.09 1.03l-4.28 4.53a.75.75 0 0 1-1.1 0L5.16 8.77a.75.75 0 0 1 .04-1.06z" />
            </svg>
          </button>
        )}
      </nav>
    </div>
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
        <Island />
        <div className="fixed top-[calc(env(safe-area-inset-top,0px)+12px)] right-3 z-50 sm:right-4">
          <ThemeToggle
            theme={theme}
            onChange={changeTheme}
            className="rounded-full border border-slate-200 bg-white/80 p-2.5 text-slate-600 shadow-sm backdrop-blur-xl transition duration-300 hover:bg-white hover:text-slate-900 dark:border-white/10 dark:bg-slate-900/80 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
          />
        </div>
        <main className="min-h-screen px-4 pt-16 pb-28 sm:px-6 md:pt-24 md:pb-8">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/engine" element={<Engine />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/replay" element={<Replay />} />
            <Route path="/module-a" element={<ModuleA />} />
            <Route path="/module-b" element={<ModuleB />} />
          </Routes>
        </main>
      </BrowserRouter>
    </ThemeContext.Provider>
  )
}
