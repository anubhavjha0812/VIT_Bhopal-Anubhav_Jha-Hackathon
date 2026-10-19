/** The numbers shown first on the Overview. Each one is copied from a stored result and names the
 *  file it came from, so a reader can check it. Tags say how far to trust the number:
 *  human = labels made by the dataset authors, design = rules were shaped on these rows,
 *  ai = labels made by AI with no human check. */
export type Tag = 'human' | 'design' | 'ai'

export interface Headline {
  label: string
  value: number
  decimals: number
  baseline: number
  baselineName: string
  note: string
  tag: Tag
  source: string
}

export const HEADLINES: Headline[] = [
  {
    label: 'Company-specific sentiment',
    value: 0.821,
    decimals: 3,
    baseline: 0.476,
    baselineName: 'headline only',
    note: 'Macro-F1 on 463 conflicting headlines, where companies in one headline have different signs.',
    tag: 'human',
    source: 'data/outputs/e27_entity_eval.json',
  },
  {
    label: 'Credit alert precision',
    value: 0.661,
    decimals: 3,
    baseline: 0.47,
    baselineName: 'rules off',
    note: 'On 243 reference pairs; recall stays 0.848. Alerts fell from 83 to 59.',
    tag: 'design',
    source: 'data/outputs/e36b_gate_precision.json',
  },
  {
    label: 'Impact vs real price moves',
    value: 0.353,
    decimals: 3,
    baseline: 0.331,
    baselineName: 'volatility alone',
    note: 'Rank match with the real two-day move on fresh 2020 stocks; 0.280 vs 0.143 in 2019.',
    tag: 'human',
    source: 'docs/others/supporting/IMPACT_V3.md',
  },
  {
    label: 'Event type on real news',
    value: 0.376,
    decimals: 3,
    baseline: 0.309,
    baselineName: 'previous engine',
    note: 'Macro-F1 over 12 classes on 200 headlines; the three credit classes have no examples in it.',
    tag: 'ai',
    source: 'data/outputs/e24_final_results.json',
  },
]

export const TAGS: Record<Tag, { text: string; hint: string }> = {
  human: { text: 'Human labels or market data', hint: 'Labels by the dataset authors, or real price moves' },
  design: { text: 'Design data', hint: 'The rules were tuned on these rows; not an unseen test' },
  ai: { text: 'AI labels', hint: 'Labels made by ChatGPT and Claude, no human check' },
}
