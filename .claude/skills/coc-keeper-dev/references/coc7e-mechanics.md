# Call of Cthulhu 7e mechanics in code

Files: `src/backend/keeper/game/coc7e/dice.py` (pure dice), `rules.py` (resolution against the sheet). Tests: `tests/test_dice.py`, `tests/test_rules.py`.

Confidence labels: **QSR** = Quick-Start Rules / Chaosium wiki (verified); **KRB\*** = Keeper Rulebook via secondary sources (medium confidence — keep such rules behind `RulesConfig` flags and mention them in the thesis limitations). The project's own rules reference (`data/open/coc7e/rules-reference.md`) should agree with the code; if you change one, check the other.

## Implemented

| Mechanic | Code | Rule | Source |
|---|---|---|---|
| d100 | `roll_d100` | tens + units; `00`+`0` = 100 | QSR |
| Success levels | `success_level` | Extreme ≤ skill//5, Hard ≤ skill//2, Regular ≤ skill; required level set by difficulty (`REQUIRED`) | QSR |
| Critical | `success_level` | roll 01 | KRB\* |
| Fumble | `success_level` + `RulesConfig.fumble_threshold_on` | 100 always; 96–100 when the number required is < 50 — `"target"` (default, difficulty-adjusted) or `"base_skill"` | KRB\* |
| Bonus/penalty dice | `net_dice`, `roll_d100` | extra tens dice, keep lowest/highest, shared units die; cancel first, then cap at `max_bonus_penalty` (2) | QSR; cap KRB\* |
| Seeded RNG | `make_rng(seed, *salt)` | `random.Random("seed:turn:0")` in evals; OS entropy in play (`seed=None`); same seed+turn → same dice in every arm | project |
| Dice expressions | `parse_expr`, `roll_expr`, `max_expr` | `1D6`, `2d6+1`, `1D4+1D4`, constants | project |
| Derived attributes | `build_investigator` | HP = (CON+SIZ)//10, MP = POW//5, SAN = POW, max SAN = 99 − Cthulhu Mythos | QSR |
| Skill / characteristic / Luck checks | `percentile_check`, `lookup` | values from the sheet; case-insensitive skill names; unknown names → error code `unknown_skill:<name>` shown to the narrator as ENGINE NOTES | QSR |
| Pushing | `can_push`, `resolve` | only the failed roll of the **immediately previous turn**, once, not after a fumble, only skill/characteristic rolls | QSR; restrictions KRB\* |
| Sanity roll | `sanity_check` | X/Y notation; success loses X, failure Y; loss clamped to current SAN; failure → `involuntary_action` | QSR |
| Fumbled SAN roll | `RulesConfig.fumbled_san_is_max_loss` | maximum loss | KRB\* |
| Temporary insanity | `sanity_check` | loss ≥ 5 in one roll → INT roll in code; **success** = temporarily insane (`temporary_insanity`) | QSR |
| Indefinite insanity flag | `sanity_check` | set on the roll where SAN lost this game day first reaches day-start SAN // 5 (`san_day_start`, `san_lost_today` on the sheet) | KRB\* |
| Game day | `start_new_day`, `resolve` | adjudicator's `new_day` (adjudicate-v3+) resets the day budget to current SAN **before** any roll that turn; draws no randomness | project |
| Permanent insanity flag | `sanity_check` | SAN reaches 0 | KRB\* |
| Scenario SAN values | `resolve(..., sanity_events=...)` | `adjudication.sanity_event` key → value from `scenario.yaml` overrides the model's `sanity_loss` | project |

Roll records (what `mechanics` stores in `roll_log` and streams as SSE `roll`): `kind, target, value, difficulty, bonus, penalty, tens, units, roll, level, passed, pushed, turn_no`; sanity adds `notation, san_loss, san_after, involuntary_action, indefinite_insanity, permanent_insanity`, optionally `temporary_insanity`; the INT check has `purpose: "temporary_insanity_check"`. Documented in `docs/api.md`.

## Not implemented (known gaps)

- **Game clock:** a new day is detected only when the *player's* action skips to a later day (adjudicate-v3 `new_day`); time skips the Keeper narrates on its own are missed, and there is no in-game date or time of day. Campaigns created on adjudicate-v2 never reset.
- **Luck spending** (KRB\* optional rule), Luck recovery.
- **Opposed rolls**, **combat** (fighting back / dodge, Extreme damage, outnumbered, manoeuvres/Build), **firearms**, **damage and HP loss**, Major Wound, dying, healing — the rules reference describes them, the engine does not; the model must currently narrate them without mechanics, which weakens invariant 1.
- **Development phase** (experience checks), **Idea/Know roll special handling**, bouts of madness tables, Cthulhu Mythos gains.
- **Difficulty from NPC skill** (Hard ≥ 50, Extreme ≥ 90).

## Adding a mechanic — pattern

1. Pure function in `dice.py`/`rules.py` returning records (dicts with the fields above plus new ones) and an updated sheet copy; no LLM imports.
2. Ambiguous rules → a `RulesConfig` flag with the literal reading as default and a comment citing the source.
3. Extend `resolve` dispatch; errors as `RuleViolation(code)` → `Resolution.errors`.
4. Extend the `Adjudication` model if the adjudicator must request it; new `adjudicate-vN` prompt describing when to use it.
5. Tests: exact threshold tables + Hypothesis invariants (e.g. a bonus die never worsens a roll; loss never exceeds current SAN).
6. Update `docs/api.md` (roll record fields), `src/frontend/src/components/RollCard.jsx`, `i18n/uk.js` labels, and the rules reference if wording differs.
7. Consider new gold adjudication items (gold-data skill) so the change is measurable.
