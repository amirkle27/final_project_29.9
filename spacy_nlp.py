"""
Lightweight spaCy utilities for common NLP tasks (NER, sentence splitting,
POS tags, lemmas, stop-words, phrase matching, and dependency viz).

If the requested model isn't installed, we fall back to a blank English
pipeline and add a sentencizer so sentence splitting still works.
"""

import spacy

from spacy.matcher import PhraseMatcher
from spacy import explain
from spacy.language import Language
from spacy import displacy


class SpacyNLP:
     """
    Thin wrapper around a spaCy pipeline with convenience helpers.

    Parameters
    ----------
    model : str, optional
        Name of a spaCy model to load (default: "en_core_web_sm").
        If loading fails, a blank English model is created with a sentencizer.
    """
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except Exception:
            # Fallback: מודל ריק + sentencizer
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def get_ents(self, text: str):
        """
        Extract named entities from free text.

        Returns
        -------
        List[(entity_text, entity_label)]
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_sentences(self, text: str):
        """
        Split text into sentences.

        Returns
        -------
        List[str]
            Sentence strings (whitespace-stripped).
        """
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents]

    def get_persons(self, sentence):
        """
        Extract PERSON entities from a sentence.

        Returns
        -------
        List[str]
            Person names as they appear in the text.
        """
        doc = self.nlp(sentence)

        return [(ent.text) for ent in doc.ents if ent.label_ == "PERSON"]

    def get_lemmas(self, sentence):
        """
        Lemmatize tokens in a sentence.

        Returns
        -------
        List[(token_text, '----->', lemma)]
            Note: the middle arrow string is kept to match the original API.
        """
        doc = self.nlp(sentence)

        return [(token.text, "----->", token.lemma_) for token in doc]

    def get_stop_words(self, sentence):
        """
        Return spaCy Token objects that are stop words (punctuation included).

        Returns
        -------
        List[spacy.tokens.Token]
        """
        doc = self.nlp(sentence)

        return [(token) for token in doc if token.is_stop]

    def get_not_stop_words(self, sentence):
        """
        Return non-stop, non-punctuation tokens as plain strings.

        Returns
        -------
        List[str]
        """
        doc = self.nlp(sentence)
        not_stop_words = []

        for token in doc:
            if not token.is_stop and not token.is_punct:
                not_stop_words.append(token.text)
        return not_stop_words

    def add_stop_words(self,sentence,word):
        """
        Mark a word as a stop word in the vocabulary, then return stop tokens
        in the provided sentence.

        Notes
        -----
        This changes global vocab state for the loaded pipeline (side-effect).
        """
        doc = self.nlp(sentence)
        self.nlp.vocab[word].is_stop = True

        return [(token) for token in doc if token.is_stop]


    def get_phrase_matcher(self, phrase, sentence):
         """
        Match exact phrases in a sentence using spaCy's PhraseMatcher.

        Parameters
        ----------
        phrase : str | Iterable[str]
            A single phrase or a collection of phrases to search for.
        sentence : str
            Text to search in.

        Returns
        -------
        List[str]
            Matched span texts, in document order.

        Notes
        -----
        Updated for spaCy v3: PhraseMatcher.add(name, patterns)
        where `patterns` is a list of Doc objects.
        """
        matcher = PhraseMatcher(self.nlp.vocab)
        matcher.add(phrase, phrase)
        doc = self.nlp(sentence)
        matches = matcher(doc)

        return [(doc[start:end].text) for match_id, start, end in matches]

    def get_tag_and_explanation(self, sentence):
        """
        Return coarse POS tag and its human-readable explanation per token.

        Returns
        -------
        List[(token_text, pos_tag, explanation)]
        """

        doc = self.nlp(sentence)

        return  [(token, token.pos_, spacy.explain(token.pos_)) for token in doc]

    def add_parser(self, parser, sentence):
        """
        Customize sentence boundaries around a delimiter-like token.

        Behavior
        --------
        For every occurrence of the exact token text `parser`,
        the following token starts a new sentence. Useful for
        custom separators (e.g., '|', '###', etc.).

        Returns
        -------
        List[str]
            Sentences with the delimiter removed and whitespace trimmed.
        """
        if "parser" in self.nlp.pipe_names:
            self.nlp.remove_pipe("parser")

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        @Language.component('set_custom_boundaries')
        def set_custom_boundaries(doc):
            for i, token in enumerate(doc[:-1]):
                if token.text == parser:
                    token.is_sent_start = False
                    doc[i + 1].is_sent_start = True
            return doc

        if "set_custom_boundaries" not in self.nlp.pipe_names:
            self.nlp.add_pipe("set_custom_boundaries", after="sentencizer")

        doc = self.nlp(sentence)

        return [(sent.text.replace(parser,"").strip())for sent in doc.sents]


    def get_tag_and_displacy(self, sentence, display_render: bool = False):
        """
        Return POS tags (with explanations) and optionally render a dependency
        visualization (Jupyter only).

        Parameters
        ----------
        display_render : bool
            If True, calls `displacy.render` with style='dep'. This is intended
            for notebooks; it won't display in a typical server context.

        Returns
        -------
        List[(token_text, pos_tag, explanation)]
        """
        doc = self.nlp(sentence)

        if display_render:
            displacy.render(doc, style="dep", jupyter=True)


        return [(token.text, token.pos_, spacy.explain(token.pos_)) for token in doc]
