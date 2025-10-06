import spacy

from spacy.matcher import PhraseMatcher
from spacy import explain
from spacy.language import Language
from spacy import displacy


class SpacyNLP:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except Exception:
            # Fallback: מודל ריק + sentencizer
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def get_ents(self, text: str):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_sentences(self, text: str):
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents]

    def get_persons(self, sentence):
        doc = self.nlp(sentence)

        return [(ent.text) for ent in doc.ents if ent.label_ == "PERSON"]



    def get_lemmas(self, sentence):
        doc = self.nlp(sentence)

        return [(token.text, "----->", token.lemma_) for token in doc]


    def get_stop_words(self, sentence):
        doc = self.nlp(sentence)

        return [(token) for token in doc if token.is_stop]


    def get_not_stop_words(self, sentence):
        doc = self.nlp(sentence)
        not_stop_words = []

        for token in doc:
            if not token.is_stop and not token.is_punct:
                not_stop_words.append(token.text)
        return not_stop_words

    def add_stop_words(self,sentence,word):
        doc = self.nlp(sentence)
        self.nlp.vocab[word].is_stop = True

        return [(token) for token in doc if token.is_stop]


    def get_phrase_matcher(self, phrase, sentence):
        matcher = PhraseMatcher(self.nlp.vocab)
        matcher.add(phrase, phrase)
        doc = self.nlp(sentence)
        matches = matcher(doc)

        return [(doc[start:end].text) for match_id, start, end in matches]



    def get_tag_and_explanation(self, sentence):
        doc = self.nlp(sentence)

        return  [(token, token.pos_, spacy.explain(token.pos_)) for token in doc]


    def add_parser(self, parser, sentence):
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
        doc = self.nlp(sentence)

        if display_render:
            displacy.render(doc, style="dep", jupyter=True)

        return [(token.text, token.pos_, spacy.explain(token.pos_)) for token in doc]