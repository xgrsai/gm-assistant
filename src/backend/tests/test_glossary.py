from pathlib import Path

from keeper.rag.glossary import Glossary

GLOSSARY = Path(__file__).resolve().parents[3] / "data" / "glossary" / "coc7e-uk-en.yaml"


def test_annotate_appends_english_terms_once():
    g = Glossary([{"en": "Sanity", "uk": [r"розсуд\w*"]}, {"en": "Luck", "uk": [r"удач\w*"]}])
    assert g.annotate("Кидаю на розсудливість, а розсудливість падає") == \
        "Кидаю на розсудливість, а розсудливість падає (Sanity)"
    assert g.annotate("Без термінів") == "Без термінів"


def test_terms_already_in_english_are_not_repeated():
    g = Glossary([{"en": "Spot Hidden", "uk": [r"помі[тч]\w*"]}])
    assert g.annotate("Кидаю Spot Hidden, щоб помітити слід") == "Кидаю Spot Hidden, щоб помітити слід"


def test_word_start_anchor():
    g = Glossary([{"en": "Hit Points", "uk": [r"хіт\w*"]}])
    assert g.terms("скільки хітів") == ["Hit Points"]
    assert g.terms("ніхто") == []


def test_project_glossary_loads():
    g = Glossary.load(GLOSSARY)
    assert g.annotate("Що дає екстремальний успіх?").endswith("(Extreme success)")
