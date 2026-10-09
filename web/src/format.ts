/** Signed number, e.g. +0.42 / −0.17 / 0.00 (never "-0.00"). */
export function signed(x: number, digits = 2): string {
  const s = x.toFixed(digits)
  if (Number(s) === 0) return (0).toFixed(digits)
  return x > 0 ? `+${s}` : s
}
