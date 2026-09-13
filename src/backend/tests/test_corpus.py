from pathlib import Path

import pytest

from keeper.citations import cited_markers, validate
from keeper.rag.corpus import chunk_id, chunk_markdown, load_corpus, split_front_matter
from keeper.rag.embeddings import PrefixedEmbeddings

REPO = Path(__file__).resolve().parents[3]

DOC = """---
source_title: Test Rules
source_type: rules
game_system: coc7e
---

# Rules

## Sanity

### Sanity Rolls

Roll against current Sanity. <!-- page: 12 -->
""" + " ".join(f"w{i}" for i in range(300)) + """

### Insanity

Losing five or more points calls for an INT roll.
"""


def test_front_matter_is_required_and_validated():
    with pytest.raises(ValueError, match="front matter"):
        split_front_matter("# no front matter")
    with pytest.raises(ValueError, match="misses"):
        split_front_matter("---\nsource_title: x\n---\ntext")
    with pytest.raises(ValueError, match="source_type"):
        split_front_matter("---\nsource_title: x\nsource_type: poetry\ngame_system: coc7e\n---\ntext")


def test_chunks_have_section_paths_overlap_and_pages():
    chunks = chunk_markdown(DOC, chunk_size=400, overlap=100)
    assert len(chunks) > 3
    assert {c.metadata["section_path"] for c in chunks} == {"Rules > Sanity > Sanity Rolls",
                                                          "Rules > Sanity > Insanity"}  # h1 differs from title
    assert all(len(c.text) <= 400 for c in chunks)
    body = [c.text for c in chunks if c.text.startswith("w")]
    assert len(body) >= 2 and body[1].split()[0] in body[0].split()  # consecutive chunks overlap
    assert all("<!--" not in c.text for c in chunks)
    assert chunks[1].metadata["page"] == 12 and chunks[0].metadata["page"] is None
    assert chunks[0].metadata["visibility"] == "public"


def test_ids_are_stable_across_line_endings_and_sources():
    a = chunk_markdown(DOC, chunk_size=400, overlap=100)
    b = chunk_markdown(DOC.replace("\n", "\r\n"), chunk_size=400, overlap=100)
    assert [c.id for c in a] == [c.id for c in b]
    assert chunk_id("Source A", "same text") != chunk_id("Source B", "same text")
    assert chunk_id("S", "a  b\n c") == chunk_id("S", "a b c")


def test_real_corpus_loads():
    chunks = load_corpus([REPO / "data" / "open" / "coc7e"], chunk_size=1000, overlap=200, root=REPO)
    types = {c.metadata["source_type"] for c in chunks}
    assert types == {"rules", "lore"}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.metadata["section_path"] for c in chunks)
    assert any(c.metadata["section_path"] == "Skill Rolls > Bonus and Penalty Dice" for c in chunks)


def test_prefixes_are_applied_once():
    class Recorder(PrefixedEmbeddings):
        pass

    seen = []

    class Inner:
        def embed_documents(self, texts):
            seen.extend(texts)
            return [[0.0] for _ in texts]

    emb = PrefixedEmbeddings(Inner(), "search_query: ", "search_document: ")
    emb.embed_query("dice")
    emb.embed_documents(["chunk"])
    assert seen == ["search_query: dice", "search_document: chunk"]


def test_citation_markers():
    assert cited_markers("A [R1]. B [R1, R3] and [R2; R4] but not [X1] or R5") == [1, 3, 2, 4]
    hits = [{"marker": 1, "chunk_id": "c1", "source": "S", "section": "A", "source_type": "rules", "score": 0.9,
             "page": None, "excerpt": "e", "text": "full"}]
    valid, invalid = validate("see [R1] and [R7]", hits)
    assert [c["chunk_id"] for c in valid] == ["c1"] and "text" not in valid[0] and invalid == [7]
