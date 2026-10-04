# Known ambiguities in the rules reference (data/open/coc7e)

Found while drafting gold data v1. Each is a place where a careful reader could label differently. Avoid building gold items on them until the corpus text is clarified; resolving one is a corpus edit (knowledge-base skill) followed by a gold-data check. Several are also worth a sentence in the thesis limitations.

| # | Topic | Sections | The ambiguity |
|---|---|---|---|
| 1 | Hard difficulty vs penalty die | Skill Rolls > Difficulty Levels; Skill Rolls > Bonus and Penalty Dice | Adverse circumstances → Hard difficulty in one section; darkness/haste → a penalty die "without changing difficulty" in the other. (All v1 adjudication items therefore leave `allowed_difficulties` empty.) |
| 2 | Clues with or without a roll | Running the Game: Keeper Principles; Common Investigator Skills > Perception Skills | "Give clues freely when investigators look in the right place" vs "searching a room thoroughly usually calls for Spot Hidden"; the scenario gates the diary behind a roll. |
| 3 | Fumble on 96–99 | Skill Rolls > Levels of Success | That 96–99 is not a fumble when the target is ≥ 50 is only implied; the project applies the threshold to the difficulty-adjusted target, the Keeper Rulebook wording refers to the number required for success. |
| 4 | Critical 01 at tiny targets | Skill Rolls > Levels of Success | Not stated whether 01 always succeeds, e.g. Locksmith 1 at Hard (half = 0). |
| 5 | Extreme attack vs dodge | Combat > Extreme Success in Combat; Combat > Melee Attacks | "The defender did not do better" vs "ties favour the dodger" when both roll Extreme. |
| 6 | First Aid on the dying | Hit Points, Wounds and Healing > Losing Hit Points; > First Aid and Medicine | Stabilising a dying character: does it also restore 1 HP and wake them? |
| 7 | Insanity durations | Sanity > Temporary Insanity; Sanity > Bouts of Madness | Temporary insanity lasts 1D10 hours; a bout lasts 1D10 rounds or hours — relation not explained. |
| 8 | "5 or more from a single Sanity roll" | Sanity > Temporary Insanity | Points are lost on the loss roll, not the Sanity roll itself. |
| 9 | Pushing INT / Idea / Know rolls | Skill Rolls > Pushing a Roll; Characteristic Rolls, Idea Rolls and Know Rolls | Not stated whether these can be pushed (notably the INT roll after SAN loss, where success is bad). |
| 10 | NPC opposition thresholds | Skill Rolls > Difficulty Levels | "Hard ≥ 50, Extreme ≥ 90" overlap; Extreme precedence only implied. |
| 11 | Opposed roll full tie | Skill Rolls > Opposed Rolls | Two alternatives (lower roll wins / stalemate) with no criterion. |
| 12 | Major Wound with odd max HP | Hit Points, Wounds and Healing > Losing Hit Points | Rounding of "half" unspecified (v1 multi-hop item uses 6 damage at 11 HP to avoid it). |
| 13 | Firearms edge cases | Combat > Firearms; Combat > Outnumbered | Beyond 4× base range undefined; point-blank given in feet; "Outnumbered" does not require several attackers. |
| 14 | Skill improvement cap | Investigator Development | "Beyond what the Keeper allows" is vague. |
| 15 | Section path prefix | lore vs rules documents | Lore paths keep the h1 prefix (`Lviv in 1927: Setting Notes > …`), rules paths do not — a `corpus.py` H1-rule quirk, not a content issue; gold v1 uses paths as they are. |
| 16 | Facts duplicated across sections | e.g. starting SAN = POW (Eight Characteristics; HP/MP/SAN/Luck), Idea = INT, "a fumble cannot be pushed" (Levels of Success; Pushing), "combat rolls cannot be pushed" (Pushing; Melee Attacks), max SAN (HP/MP/SAN/Luck; Cthulhu Mythos), DEX order (Eight Characteristics; Combat Rounds) | Non-multi-hop items list one primary section, so Recall can be slightly understated when the retriever returns the other one. Consider listing both sections for such items in v2. |

## Borderline v1 items flagged during drafting

- rules_qa: qa-030 (a forced push could mean a different skill), qa-075 (clues freely), qa-031 and qa-064 (facts also elsewhere); unanswerable qa-086 (chases partly touched), qa-091 (Credit Rating), qa-088 (.38 damage — general damage scale invites a ruling), qa-098 (mayor of Lviv — a model may know it from memory).
- conflict_probes: complex edits cp-004, cp-010, cp-020; weaker plausibility cp-022, cp-024; familiarity labels for cp-008 and cp-010 are judgement calls.
- adjudication: single target where another skill is defensible — g19 (Library Use vs Spot Hidden), g22 (Persuade vs Charm), g24 (Charm vs Persuade), g29 (Occult vs Know/EDU), g23 (Fast Talk); roll-or-not debatable — g18, g32, g38; ordering questions — g33 (SAN first?), g40 (SAN belonged to the previous turn?), g37 (INT roll phrased partly out of character); g01 (lifting the rug without a roll is per scenario, but Spot Hidden is tempting).
