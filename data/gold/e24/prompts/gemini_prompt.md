# E24 prompt for Gemini (labeller B)

How to use (record every item below in the annotation record; do not guess any of them):
- Interface: the Gemini web app or Google AI Studio / the Gemini API, whichever you actually use.
- Start a NEW chat (Gemini Apps activity and saved info off if available), so earlier chats cannot
  leak in.
- Paste everything from the instructions line to the end of GUIDE.md, then paste the input rows
  (`item_id,text,date,companies`, copied from the blind file) in batches of at most 50.
- Never paste engine predictions, ChatGPT's output, or any evaluation result.
- Model and version: write exactly what the interface shows (for example the model picker name);
  if it shows no version, write "not shown". Do not invent one.
- Settings: the web app does not expose temperature or a seed, so record "web default, not
  controllable"; with AI Studio or the API record the temperature you set.
- Reproducibility: provider models change without notice; a rerun may give other labels. Keep the
  raw outputs exactly as returned.

=== E24 ANNOTATION INSTRUCTIONS (identical for every labeller) ===
You are labelling financial news headlines and tweets for a research test set.
The full annotation guide (GUIDE.md) is pasted below these instructions; follow it exactly.

For each input row, choose EXACTLY ONE label from this list, spelled exactly as written:
Default | Credit downgrade | Covenant breach | Geopolitical | Macroeconomic | M&A | Product news | Earnings | Analyst rating | Legal/Regulatory | Management change | Other

Rules:
1. Label the main news fact the text reports (primary-event rule in the guide). Use the guide's
   overlap table only for the overlaps it lists.
2. Use only what the text itself says. Do not infer an event the text does not report. Do not
   use outside knowledge of what happened later, and do not look anything up.
3. "Product news" is narrow: a launch, unveiling, release, recall, or a product deal. Other
   company news is "Other" unless another class fits.
4. Set ambiguous to Y when two labels fit equally well or the text cannot be judged; still give
   your best single label.
5. The rationale is at most 20 words and points to the words in the text that support the label.
6. Label every row you are given, in the same order, once. Do not skip, merge or add rows.

Output: CSV only, no other text, with this exact header line and one line per input row:
item_id,label,ambiguous,rationale
- item_id: copied exactly from the input
- label: one of the 12 labels above
- ambiguous: Y or N
- rationale: wrapped in double quotes
=== END OF INSTRUCTIONS ===
