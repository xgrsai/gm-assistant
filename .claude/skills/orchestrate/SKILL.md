---
name: orchestrate
description: Carry a GM Assistant task from idea to a reviewed change by delegating to the project agents — planner (plan), implementer (one step at a time), reviewer (independent check) — with the author approving the plan and every commit. Use when the user invokes /orchestrate <task>, asks to do something "через агентів", «оркестратор», «доведи до кінця», «зроби фічу від плану до рев'ю», or wants to finish a roadmap item (judge calibration, a new mechanic, Phase 4 LoRA prep) end to end.
---

# Orchestrating planner → implementer → reviewer

The main session is the orchestrator: subagents cannot start other subagents, so this skill runs here and calls the agents in `.claude/agents/` with the Agent tool (`subagent_type`: `planner`, `implementer`, `reviewer`). The orchestrator keeps the conversation with the author, holds the plan, decides what runs next and is the only one that commits.

## 1. Frame the task

- Take the task from the arguments. If it is "what's left" / «докінчити», read the roadmap and open items in `.claude/CLAUDE.md` and `git log --oneline -15`, propose 3-5 candidate tasks with a one-line value/cost each, and let the author pick.
- Check `git status`: start from a clean tree (or say what is uncommitted and ask whether it belongs to the task).

## 2. Plan

- Call **planner** with: the task, constraints the author stated, and anything already decided in this conversation. Ask for the plan format from its definition.
- Read the plan critically: do the named files exist, are steps small, are reproducibility effects (prompt versions, dataset hashes, eval fingerprints) and costs (Gemini calls, downloads) stated?
- Show the author a short version (goals, steps, decisions, cost) and **wait for approval**. Questions in «Рішення автора» are asked now, with the planner's recommendation first. Skip the approval gate only for a single-step, no-cost change the author already specified exactly.

## 3. Implement, one step at a time

- For each step call **implementer** with: the step text verbatim, the relevant part of the plan (goal, decisions, dependencies), and what earlier steps changed. One step per call.
- Steps marked parallel with disjoint files may run as parallel implementer calls; everything else runs in order.
- If the implementer reports a deviation or a blocker, decide: adjust the plan yourself if it is mechanical, re-plan with the planner if the design changed, or ask the author if it is a real decision.

## 4. Review

- After each step (or a small group of related steps) call **reviewer** with: the step(s) and the diff scope (`git diff`, or the files involved).
- CHANGES REQUESTED → send the blocker/major findings verbatim to the implementer, then review again. At most 2 fix rounds; after that, stop and show the author the open findings.
- Do not accept a finding blindly either: if it looks wrong, check the code yourself before forwarding it.

## 5. Verify and hand back

- Run the offline checks yourself at the end: `uv run ruff check .` and `uv run pytest -q` in `src/backend`, `npm run build` in `src/frontend` if touched. Report results faithfully, with output on failure.
- Anything that needs live Gemini calls, eval runs, ingestion, model downloads or long local runs: describe the exact command and cost and let the author run or approve it — never start it on your own.
- Commit only when the author asks: one commit per step (code + tests + docs), conventional message (`feat(...)`, `fix(...)`, …) ending with the session's attribution line. Never push without being asked.

## Report to the author (Ukrainian)

What was done per step, review verdicts and what was fixed after review, check results, what is left or needs their decision, and the suggested commit(s).
