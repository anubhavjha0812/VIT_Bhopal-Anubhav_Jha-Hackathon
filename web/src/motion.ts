import gsap from 'gsap'
import { useGSAP } from '@gsap/react'

gsap.registerPlugin(useGSAP)

/** Every animation runs inside this query, so a viewer who asks for less motion sees the
 *  final state at once. */
export const MOTION_OK = '(prefers-reduced-motion: no-preference)'

export { gsap, useGSAP }
