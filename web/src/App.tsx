import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import Intro from './components/Intro'
import ThemeToggle from './components/ThemeToggle'
import Analyze from './pages/Analyze'
import Engine from './pages/Engine'
import ModuleA from './pages/ModuleA'
import ModuleB from './pages/ModuleB'
import Overview from './pages/Overview'
import Replay from './pages/Replay'
import { shouldShowIntro } from './intro'
import { applyTheme, initialTheme, ThemeContext, type Theme } from './theme'

// A page gets a link here only once it exists (no empty sections).
const PAGES = [
  { path: '/', label: 'Overview' },
  { path: '/engine', label: 'Engine' },
  { path: '/analyze', label: 'Analyze' },
  { path: '/replay', label: 'Live replay' },
  { path: '/module-a', label: 'Module A' },
  { path: '/module-b', label: 'Module B' },
]

const EASE = 'cubic-bezier(0.32, 0.72, 0, 1)' // slow, soft landing (iOS sheet curve)
const GROW_MS = 650
const OPEN_DELAY_MS = 120 // a mouse passing over does not open it
const CLOSE_DELAY_MS = 280

/** Floating page switcher: a small pill with the current page that opens into every page. The
 *  theme toggle sits at its right end in both states. Top centre on wide screens, bottom on phones.
 *
 *  Both faces sit centred on top of each other and the pill animates its exact size between them,
 *  growing from the centre while the closed face fades out and the links fade in one by one. */
function Island({ theme, onTheme, ready }: { theme: Theme; onTheme: (t: Theme) => void; ready: boolean }) {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const [sizes, setSizes] = useState<Record<'closed' | 'open', [number, number]> | null>(null)
  const closedFace = useRef<HTMLDivElement>(null)
  const openFace = useRef<HTMLDivElement>(null)
  const timer = useRef<number | undefined>(undefined)
  const current = PAGES.find((p) => p.path === pathname) ?? PAGES[0]

  // Once the page scrolls, the closed island turns to frosted glass so the content shows through.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // The size of each face, kept current as fonts load or the page name changes.
  useLayoutEffect(() => {
    const size = (el: HTMLDivElement | null): [number, number] => [el?.offsetWidth ?? 0, el?.offsetHeight ?? 0]
    const measure = () =>
      setSizes({
        closed: size(closedFace.current),
        open: size(openFace.current),
      })
    measure()
    const ro = new ResizeObserver(measure)
    for (const el of [closedFace.current, openFace.current]) if (el) ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => () => window.clearTimeout(timer.current), [])

  // After the intro, the island appears and opens once on its own to show every page.
  const hidden = useRef(!ready)
  useEffect(() => {
    if (!ready || !hidden.current) return
    const show = window.setTimeout(() => setOpen(true), 700)
    const hide = window.setTimeout(() => {
      hidden.current = false // once only (set here, not above, so a re-run effect still peeks)
      setOpen(false)
    }, 2600)
    return () => [show, hide].forEach((t) => window.clearTimeout(t))
  }, [ready])
  const later = (next: boolean, ms: number) => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setOpen(next), ms)
  }
  const glass = scrolled && !open
  const fade = (shown: boolean, delay = 0) => ({
    opacity: shown ? 1 : 0,
    filter: shown ? 'none' : 'blur(4px)',
    transform: shown ? 'none' : 'scale(0.96)',
    transition: `opacity 380ms ${EASE} ${delay}ms, filter 380ms ${EASE} ${delay}ms, transform 480ms ${EASE} ${delay}ms`,
  })

  const toggle = (
    <>
      <span className="mx-1 h-4 w-px shrink-0 bg-white/15 dark:bg-slate-900/15" aria-hidden />
      <ThemeToggle
        theme={theme}
        onChange={onTheme}
        className="shrink-0 rounded-full p-1.5 text-white/70 transition duration-300 hover:bg-white/10 hover:text-white dark:text-slate-600 dark:hover:bg-slate-900/10 dark:hover:text-slate-900"
      />
    </>
  )

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-[calc(env(safe-area-inset-bottom,0px)+12px)] z-50 flex justify-center px-3 md:top-[calc(env(safe-area-inset-top,0px)+12px)] md:bottom-auto">
      <nav
        aria-label="Pages"
        onMouseEnter={() => later(true, OPEN_DELAY_MS)}
        onMouseLeave={() => later(false, CLOSE_DELAY_MS)}
        // keyboard focus opens it; a tap on the theme toggle should not
        onFocus={(e) => e.target.matches(':focus-visible') && later(true, 0)}
        onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && later(false, 0)}
        onKeyDown={(e) => e.key === 'Escape' && later(false, 0)}
        style={{
          width: sizes?.[open ? 'open' : 'closed'][0],
          height: sizes?.[open ? 'open' : 'closed'][1],
          maxWidth: 'calc(100vw - 24px)',
          opacity: ready ? 1 : 0,
          transform: ready ? 'none' : 'translateY(-6px) scale(0.9)',
          transition: ['width', 'height', 'background-color', 'backdrop-filter', 'box-shadow', 'opacity', 'transform']
            .map((k) => `${k} ${k === 'width' || k === 'height' ? GROW_MS : 700}ms ${EASE}`)
            .join(', '),
        }}
        // always the opposite of the page: dark on the light theme, off-white on the dark theme
        className={`island pointer-events-auto relative overflow-hidden rounded-[28px] border border-white/10 text-white shadow-xl shadow-slate-900/25 dark:border-white/60 dark:text-slate-900 dark:shadow-black/60 ${
          open ? 'shadow-2xl' : ''
        } ${glass ? 'bg-slate-950/35 backdrop-blur-2xl dark:bg-slate-100/80' : 'bg-slate-950/90 backdrop-blur-xl dark:bg-slate-100/95'}`}
      >
        {/* a teal light circles the border while the island is closed (index.css) */}
        <div
          className="island-ring"
          style={{ opacity: open ? 0 : 1, transition: `opacity 500ms ${EASE}` }}
          aria-hidden
        />
        {/* each face is centred, so a face wider than the pill overflows both sides evenly */}
        <div className={`absolute inset-0 flex items-center justify-center ${open ? 'pointer-events-none' : ''}`}>
          <div
            ref={closedFace}
            aria-hidden={open}
            className="flex w-max shrink-0 items-center pr-1.5"
            style={fade(!open, open ? 0 : 160)}
          >
            <button
              type="button"
              tabIndex={open ? -1 : 0}
              onClick={() => later(true, 0)}
              aria-expanded={open}
              className="flex items-center gap-2.5 py-2 pr-2 pl-3 text-sm whitespace-nowrap"
            >
              <span className="h-2 w-2 shrink-0 rounded-full bg-teal-400 shadow-[0_0_8px] shadow-teal-400 dark:bg-teal-600 dark:shadow-teal-500/70" />
              <span className="text-white/60 dark:text-slate-500">AI Risk Engine</span>
              <span className="font-semibold">{current.label}</span>
              <svg
                viewBox="0 0 20 20"
                className="h-4 w-4 text-white/50 dark:text-slate-400"
                fill="currentColor"
                aria-hidden
              >
                <path d="M5.2 7.7a.75.75 0 0 1 1.06.04L10 11.7l3.74-3.96a.75.75 0 1 1 1.09 1.03l-4.28 4.53a.75.75 0 0 1-1.1 0L5.16 8.77a.75.75 0 0 1 .04-1.06z" />
              </svg>
            </button>
            {toggle}
          </div>
        </div>
        <div className={`absolute inset-0 flex items-center justify-center ${open ? '' : 'pointer-events-none'}`}>
          <div ref={openFace} aria-hidden={!open} className="flex w-max shrink-0 items-center p-1.5">
            <ul className="no-scrollbar flex min-w-0 gap-1 overflow-x-auto whitespace-nowrap">
              {PAGES.map((p, i) => (
                <li key={p.path} style={fade(open, open ? 140 + 35 * i : 0)}>
                  <NavLink
                    to={p.path}
                    end
                    tabIndex={open ? 0 : -1}
                    // a mouse click keeps it open until the pointer leaves, like the theme toggle;
                    // a tap or the keyboard (no hover to leave) closes it
                    onClick={(e) => (e.detail === 0 || !window.matchMedia('(hover: hover)').matches) && later(false, 0)}
                    className={({ isActive }) =>
                      `block rounded-full px-3.5 py-1.5 text-sm transition-colors duration-300 ${
                        isActive
                          ? 'bg-white font-semibold text-slate-900 dark:bg-slate-900 dark:text-white'
                          : 'text-white/75 hover:bg-white/10 hover:text-white dark:text-slate-600 dark:hover:bg-slate-900/10 dark:hover:text-slate-900'
                      }`
                    }
                  >
                    {p.label}
                  </NavLink>
                </li>
              ))}
            </ul>
            <div className="flex items-center" style={fade(open, open ? 140 + 35 * PAGES.length : 0)}>
              {toggle}
            </div>
          </div>
        </div>
      </nav>
    </div>
  )
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [intro, setIntro] = useState(shouldShowIntro)
  const endIntro = useCallback(() => setIntro(false), [])
  const changeTheme = (t: Theme) => {
    applyTheme(t)
    setTheme(t)
  }

  return (
    <ThemeContext.Provider value={theme}>
      <BrowserRouter>
        {intro && <Intro onDone={endIntro} />}
        <Island theme={theme} onTheme={changeTheme} ready={!intro} />
        <main className="min-h-screen px-4 pt-6 pb-28 sm:px-6 md:pt-[4.5rem] md:pb-8">
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
