---
name: knowledge-base
description: Add, edit or re-ingest texts in the GM Assistant RAG knowledge base (data/open, data/private → Qdrant) — rules references, setting lore, scenario or narrative-example documents, front matter, heading structure, chunking, embeddings, collections, and checking what retrieval returns. Use this whenever the user wants to extend or fix the corpus, add a new rule/lore section or a PDF/Markdown source, change chunk size or embedder, rebuild a collection, debug why the Keeper cites the wrong passage, or asks whether some book text may be put into the repo — including Ukrainian requests like «додай у базу знань», «оновити корпус», «переіндексуй», «чому пошук не знаходить», «додай правила про погоню».
---

# Maintaining the knowledge base

The corpus is the "R" in RAG: the Keeper's rules answers are only as good as these texts and how well they chunk. It also anchors the evaluation — gold labels point at section headings — so corpus edits ripple into evals. Work through the steps in order.

## 1. Decide where the text may live (copyright first)

- `data/open/<system>/` — committed, MIT. Only text written for this project or genuinely open (public domain, ORC/CC licences with attribution). Mechanics may be summarised in original words; book prose may not be copied.
- `data/private/<system>/` — git-ignored. Anything from purchased or copyrighted books (Chaosium rulebooks, published scenarios) and anything derived from them. Never move it into `data/open`, never quote it into prompts, tests, gold data or commits.
- Also: free-tier Gemini keys may use prompts for product improvement, so private rulebook text should only reach Gemini through a paid key. Mention this if private text is being ingested for play.

If the user wants to add text from a book, ask which licence applies before writing anything.

## 2. Write or edit the document

Each source is one Markdown file with YAML front matter and a heading hierarchy. The rules that make sections retrievable and eval-friendly (one topic per `###` section, self-contained wording, ~≤900 characters so a section stays one chunk, stating terms the way players ask about them) are in `references/writing-guide.md`. Front matter fields, allowed `source_type` values and how section paths are derived: `references/format.md`. Read both before writing a new document.

For PDFs: convert offline to Markdown (e.g. Docling in a separate environment — it is not a backend dependency), keep `<!-- page: N -->` markers so citations get page numbers, put the result in `data/private/`.

## 3. Check the impact on evaluation before re-ingesting

Gold data (`data/eval/<system>/*.jsonl`) references sections by `source_title` + heading path, and conflict probes reference exact substrings of a chunk.
- Renaming or moving a heading, or changing `source_title`, breaks every gold label pointing at it.
- Editing a sentence used by a conflict probe's `find` breaks that probe.
- Growing a section past one chunk breaks probes on it ("section spans several chunks").

Run `uv run python -m evals.datasets` (from `src/backend`) after the edit. If it reports problems, tell the user: frozen gold files are never edited in place — the gold-data skill describes creating a v2. Adding new sections is safe.

## 4. Ingest

From `src/backend` (Qdrant must be running: `docker compose up -d qdrant`; Ollama with `nomic-embed-text`):

```
uv run python -m keeper.rag.ingest ../../data/open/coc7e --recreate
uv run python -m keeper.rag.ingest ../../data/open/coc7e ../../data/private/coc7e --recreate   # open + private
```

- The collection name encodes system, embedder and chunking (`coc7e__nomic15__c1000o200`). `--chunk-size`, `--overlap` or `--embedder` create a *different* collection, so experiments on the old one stay reproducible.
- `--recreate` drops and rebuilds that one collection. Without it, upserts leave stale chunks from deleted text behind — use `--recreate` after deletions or heading changes.
- Output: chunk counts per source type and a dump at `artifacts/corpus/<collection>/chunks.jsonl` (exact texts, ids, section paths — use it to write gold data).
- A changed corpus changes retrieval results; campaigns and eval runs record the collection name, but not its content. Tell the user when a re-ingest makes earlier eval numbers non-comparable.

## 5. Verify retrieval

```
uv run python -m keeper.rag.search "how do bonus dice work"
uv run python -m keeper.rag.search "sanity loss for seeing a corpse" --arm play-gemini-rag
```
Try the questions a player would ask, in English (the analyze node translates Ukrainian turns). The new section should come first; scores below the arm's `min_score` (0.55 now) are dropped in play. For systematic numbers run the retrieval task (eval-run skill).

## Other configuration

Embedder profiles, quotas, `min_score`, source-type filters, adding a new game system or an embedder like bge-m3: `references/format.md` § Configuration.
