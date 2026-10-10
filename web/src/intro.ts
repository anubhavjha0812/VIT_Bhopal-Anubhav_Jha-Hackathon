/** Show the intro (components/Intro.tsx) on every page load, unless the viewer prefers less
 *  motion. Moving between pages does not reload, so it plays on opening or refreshing only. */
export function shouldShowIntro(): boolean {
  return !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}
