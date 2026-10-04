# Corpus format and configuration

Code: `src/backend/keeper/rag/corpus.py` (parsing, chunking, ids), `embeddings.py`, `retriever.py`, `ingest.py`, `search.py`. Config: `src/backend/config/embedders.yaml`, retrieval blocks in `src/backend/experiments/*.yaml` (play) and `src/backend/evals/experiments/*.yaml` (evals).

## Contents
1. Front matter
2. Headings and section paths
3. Chunking and ids
4. What retrieval returns
5. Configuration (embedders, quotas, min_score, new systems)

## 1. Front matter

```markdown
---
source_title: Call of Cthulhu 7e Rules Reference (GM Assistant)   # required; shown in citations; part of chunk ids
source_type: rules            # required: rules | lore | scenario | narrative_example
game_system: coc7e            # required; retrieval filters on it
visibility: public            # optional, default public; play retrieval only reads public
language: en                  # optional, default en
license: MIT (original text written for this project)           # optional but expected
---
```

A file without front matter, or with a missing required field or unknown `source_type`, fails ingestion.

- `rules` and `lore` are retrieved in play today (quotas `rules: 4, lore: 1`).
- `scenario` (secret Keeper-only material, use `visibility: keeper`) and `narrative_example` (style samples) are accepted by the parser but not yet used by prompts — adding them requires quota and prompt changes (coc-keeper-dev skill).

## 2. Headings and section paths

Split levels: `#` (h1), `##` (h2), `###` (h3). A chunk's `section_path` joins its headings with ` > `.

**H1 rule:** if `source_title` starts with the h1 text, the h1 is dropped from the path (it adds nothing). Current corpus:
- rules: title "Call of Cthulhu 7e Rules Reference (GM Assistant)", h1 "Call of Cthulhu 7e Rules Reference" → paths like `Skill Rolls > Bonus and Penalty Dice`.
- lore: title "Lviv 1927 Setting Notes (GM Assistant)", h1 "Lviv in 1927: Setting Notes" → the h1 is NOT a prefix of the title, so paths start with `Lviv in 1927: Setting Notes > ...`. Gold data v1 uses these paths as they are; don't "fix" the lore title or h1 without creating gold data v2.

For new documents, make the h1 equal the beginning of `source_title` to get clean paths.

Text before the first `##` becomes a section whose path is the h1 (or empty after the H1 rule → the h1 itself is kept).

## 3. Chunking and ids

- Per section: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` (defaults of `ingest`), headings kept in the text (`strip_headers=False`), so a chunk starts with its `###` line.
- `<!-- page: N -->` markers are removed from chunk text; the page in effect at the chunk start is stored as `page`.
- Chunk id = `uuid5(namespace, sha256(source_title + "\x1f" + normalised text))`; normalisation = NFC + collapsed whitespace. Same text → same id on any OS; identical text in two sources → different ids. Duplicate text inside one source is dropped.
- Payload layout matches langchain-qdrant: `{"page_content": text, "metadata": {source_title, source_type, game_system, visibility, language, license, section_path, page, source_path, chunk_index}}`.

## 4. What retrieval returns

`Retriever.search(query, RetrievalSpec, game_system=...)` (play and frozen eval contexts):
- embeds the query with the embedder's query prefix (`search_query: ` for nomic; documents get `search_document: `; prefixes are added by composition — never subclass `OllamaEmbeddings`, it double-prefixes queries);
- for each `source_type` in `quotas`: exact cosine search filtered by `game_system`, `source_type`, `visibility=public`, `score_threshold=min_score`, `limit=quota`;
- concatenates results in quota order and numbers them `marker` 1..n → the Keeper cites `[R1]`, `[R2]`.

`Retriever.rank(...)` returns one ranked list over all types without quotas or threshold (retrieval evaluation).

Startup (`server/main.py`) and eval runs verify that every RAG arm's collection exists, has the embedder's vector size, and that the embedder answers.

## 5. Configuration

**Embedder profile** (`config/embedders.yaml`):
```yaml
nomic15:
  provider: ollama
  model: nomic-embed-text     # English-only, 768 dims
  dim: 768
  query_prefix: "search_query: "
  document_prefix: "search_document: "
```
Adding e.g. `bge-m3` (multilingual, 1024 dims, no prefixes): `ollama pull bge-m3` (ask first — it is a ~1 GB download), add a profile, ingest with `--embedder bgem3` into its own collection, create arms/experiments whose `retrieval.embedder` and `collection` point to it. Raw Ukrainian retrieval with nomic scored Hit@5 0.22 vs 0.92 for English, so a multilingual embedder is a meaningful ablation.

**Retrieval spec** (per arm, frozen per campaign; changing it creates a new spec hash):
```yaml
retrieval:
  collection: coc7e__nomic15__c1000o200
  embedder: nomic15
  quotas: {rules: 4, lore: 1}
  exact: true          # exhaustive search: deterministic
  min_score: 0.55      # calibrated: 0.55–0.60 keeps ≥95% recall on English queries
```
In play, change `src/backend/experiments/play.yaml` by adding a new arm rather than editing the existing arm, so old campaigns keep a spec that matches what they ran with.

**New game system:** new `data/open/<system>/` with `game_system: <system>` in front matter, a separate collection, a scenario pack with `game_system: <system>`, and (for mechanics) a `keeper/game/<system>` module — see coc-keeper-dev skill. Retrieval filters by the scenario's `game_system`.
