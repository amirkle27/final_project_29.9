"""
Integration test for the `spacy_nlp` module.

This test verifies that the SpaCy NLP wrapper (`SpacyNLP`) functions correctly,
including:
- Named Entity Recognition (NER)
- Sentence segmentation
- Part-of-speech tagging

It automatically skips the test if SpaCy or its language model (`en_core_web_sm`)
is not installed.
"""

import pytest

def test_spacy_basic_ents_and_tags():
    """
    Test basic SpaCy entity recognition, sentence splitting, and tagging.

    Steps:
        1. Attempt to import SpaCy (skip test if unavailable).
        2. Attempt to load the English model (`en_core_web_sm`).
           - Skip the test if the model is not installed.
        3. Initialize the custom `SpacyNLP` wrapper class.
        4. Run NER on a sample text and verify:
            - It identifies at least one `PERSON` or `ORG` entity.
            - It identifies at least one `GPE` or `LOC` entity.
        5. Verify sentence segmentation returns multiple sentences.
        6. Verify grammatical tagging produces at least one token tag.
    """
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

    # sentences test
    sents = s.get_sentences(text)
    assert len(sents) >= 2

    # grammer tags test
    tags = s.get_tag_and_displacy(text, display_render=False)
    assert len(tags) > 0



