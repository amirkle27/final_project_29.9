import pytest

def test_spacy_basic_ents_and_tags():
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("SpaCy model en_core_web_sm is not installed")

    from spacy_nlp import SpacyNLP
    s = SpacyNLP()

    text = "Barack Obama visited Paris in 2015. Apple unveiled a new iPhone in California."
    ents = s.get_ents(text)
    labels = [lbl for _, lbl in ents]
    assert any(lbl in ("PERSON", "ORG") for lbl in labels)
    assert any(lbl in ("GPE", "LOC") for lbl in labels)

    # בדיקת משפטים
    sents = s.get_sentences(text)
    assert len(sents) >= 2

    # בדיקת תגיות דקדוקיות
    tags = s.get_tag_and_displacy(text, display_render=False)
    assert len(tags) > 0

