# M1: Module B sensitivity checks

Rules fixed 2026-10-11, before the checks are written or run. Written down before the experiment was run; unchanged here.

## Rules

Consistency checks on the real book (`data/reference/book.csv`) and market data; they change
no result. Expected behaviour, fixed now. If a check fails, the formula is investigated; the
expectation is not changed to fit the result.

1. **No shock, no loss:** no obligor, `multiple = 0` → loss 0, CET1 after = before, every
   rating, spread and PD unchanged.
2. **Bigger market shock, bigger loss:** for each episode (GFC 2008, euro 2011, COVID 2020) and
   `multiple` 0, 0.5, 1, 1.5, 2 (no obligor): total loss does not fall and CET1 after does not
   rise as the multiple grows; Baa spread after and z grow with it.
3. **`multiple = 0` removes the episode:** with a downgrade, Aaa and Baa stay at their book-date
   levels, z = 0 and USD/INR does not move; only the obligor's rating changes, and every other
   name's spread moves by its contagion term only (≥ 0).
4. **More notches, more credit risk:** one obligor, `multiple = 0`, 0 to 6 notches: its PD after,
   its revolver ECL, the bank's total loss and RWA after do not fall as notches grow.
5. **Stressed PD:** Vasicek PD rises with z, and is at least the normal one-year PD in the
   scenario (z > 0).
6. **Default:** `notches = 99` → rating D, revolver in stage 3 with ECL = LGD × EAD, bond worth
   notional × (1 − LGD).
7. **Reverse stress:** at the multiple it returns, CET1 is at the 8% floor (within 0.05 point),
   and at 90% of that multiple CET1 is above the floor.

## Result

- **Question:** Module B sensitivity: do the formulas move the right way?
- **Outcome:** 7 checks (13 tests) on the real book: no shock → no loss; loss grows with the episode multiple (3 episodes); multiple 0 removes the episode; more notches → more PD/ECL/loss/RWA; stressed PD rises with z; default → stage 3, bond at recovery; reverse stress lands on 8%
- **Verdict:** all pass
- **Stored results:** `tests/test_module_b_sensitivity.py`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
