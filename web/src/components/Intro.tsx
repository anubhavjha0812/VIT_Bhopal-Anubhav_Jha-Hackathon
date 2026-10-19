import { useEffect, useState } from 'react'

const EXIT_AT_MS = 4600 // everything is on screen by 2.2 s, then held ~2.4 s so it can be read
const EXIT_MS = 700

/** ~5.3 s opening: the island's dot lights up, a ring of light circles it, the title and the
 *  hackathon line rise in, then everything shrinks up into the island. Any click or key skips. */
export default function Intro({ onDone }: { onDone: () => void }) {
  const [leaving, setLeaving] = useState(false)

  useEffect(() => {
    const leave = () => setLeaving(true)
    const exit = window.setTimeout(leave, EXIT_AT_MS)
    window.addEventListener('keydown', leave)
    return () => {
      window.clearTimeout(exit)
      window.removeEventListener('keydown', leave)
    }
  }, [])

  useEffect(() => {
    if (!leaving) return
    const done = window.setTimeout(() => {
      document.documentElement.style.background = '' // set dark in index.html for the first paint
      onDone()
    }, EXIT_MS)
    return () => window.clearTimeout(done)
  }, [leaving, onDone])

  const rise = (delay: number) => ({ animation: `intro-rise 900ms cubic-bezier(0.32,0.72,0,1) ${delay}ms both` })
  return (
    <div
      onClick={() => setLeaving(true)}
      className="fixed inset-0 z-[60] flex cursor-pointer items-center justify-center bg-slate-950 text-white"
      style={{ opacity: leaving ? 0 : 1, transition: `opacity ${EXIT_MS}ms cubic-bezier(0.32,0.72,0,1)` }}
      role="presentation"
    >
      <div
        className="flex flex-col items-center px-6 text-center"
        style={{
          transform: leaving ? 'translateY(-40vh) scale(0.45)' : 'none',
          filter: leaving ? 'blur(2px)' : 'none',
          transition: `transform ${EXIT_MS}ms cubic-bezier(0.32,0.72,0,1), filter ${EXIT_MS}ms`,
        }}
      >
        <div className="relative mb-8 h-16 w-16 rounded-full" style={{ animation: 'intro-pop 700ms cubic-bezier(0.32,0.72,0,1) both' }}>
          <div className="island-ring" style={{ animationDuration: '2.4s', animationDelay: '350ms' }} />
          <span className="absolute inset-0 m-auto h-3 w-3 rounded-full bg-teal-400 shadow-[0_0_16px_4px] shadow-teal-400/70" />
        </div>
        <h1 className="text-4xl font-semibold tracking-tight sm:text-6xl" style={rise(800)}>
          AI Risk Engine
        </h1>
        <p className="mt-4 text-xs font-medium text-teal-300/90 sm:text-sm" style={rise(2600)}>
          S&amp;P Global × Crisil · Campus Hackathon 2026
        </p>
        <p className="mt-2 text-sm text-white/50" style={rise(2000)}>
          Anubhav Jha · VIT Bhopal
        </p>
      </div>
      <p className="absolute bottom-8 text-[11px] tracking-wide text-white/30" style={rise(2600)}>
        click or press any key to skip
      </p>
    </div>
  )
}
