export const SEEN_KEY = 'intro-seen'

/** Show the intro (components/Intro.tsx) on the first visit of a session, unless the viewer
 *  prefers less motion. */
export function shouldShowIntro(): boolean {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return false
  try {
    return sessionStorage.getItem(SEEN_KEY) == null
  } catch {
    return true // storage blocked: still show it once per page load
  }
}
