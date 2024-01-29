import os
import numpy as np
import scipy.sparse as sp

from Orange.data import Table
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus

import spacy
import pandas as pd

import storynavigation.modules.util as util


class OWSNSettingAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = '3) Setting Analysis'
    description = "Provides tools to help identify the main setting of a story."
    icon = "icons/setting_analysis_icon.png"
    priority = 6480

    NL_SPACY_MODEL = "nl_core_news_sm" 

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        # edge_data = Output('Edge Data', Table)
        # TODO: add output here 
        pass 


    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.prominence_scores = {}
        self.corpus = None # initialise list of documents (corpus)
        self.nlp_nl = None # initialise spacy model

    def load_spacy_pipeline(self, name): # TODO: can we take this from util?
        """Check if the spacy language pipeline was downloaded and load it.
        Downloads the language pipeline if not available.
    
        Args:
            name (string): Name of the spacy language.
    
        Returns:
            spacy.language.Language: The spacy language pipeline
        """
        if spacy.util.is_package(name):
            nlp = spacy.load(name)
        else:
            os.system(f"spacy download {name}")
            nlp = spacy.load(name)
        return nlp

    @Inputs.corpus
    def set_data(self, corpus=None):
        self.nlp_nl = self.load_spacy_pipeline(self.NL_SPACY_MODEL)
        self.nlp_nl.add_pipe("merge_noun_chunks")
        self.corpus = corpus
        # self._generate_network(self.corpus)
        self.get_embeddings()

    def reset_widget(self):
        self.corpus = None
        self.Warning.clear()


    def get_embeddings(self):
        """
        Processes the corpus and extracts embeddings for selected words.
        """
        idx = 0 # TODO: replace this when #32 is merged
        self.stories = []
        for _ in self.corpus:
            text = ''
            for field in self.corpus.domain.metas:
                text_field_name = str(field)
                if text_field_name.lower() in ['text', 'content']:
                    text = str(self.corpus[idx, text_field_name])

            if len(text) > 0:
                self.stories.append((text, idx))

            idx += 1
        
        # process one story (like __process_story in tagging). TODO: expand to all. see __process_stories
        story_text, story_id = self.stories[0]
        sentences = util.preprocess_text(story_text)

        # generate and store nlp tagged models for each sentence
        tagged_sentences = []
        for sentence in sentences:
            if (len(sentence.split()) > 0): # sentence has at least one word in it
                tagged_sentence = self.nlp_nl(sentence) # TODO: what is a good way to store the nlp model here? -- see other code. also, use nl/en model as necessary
                tagged_sentences.append(tagged_sentence)
        
        first_sentence = tagged_sentences[0]
        lemmatized = [token.lemma_ for token in first_sentence]
        breakpoint()



if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    corpus_ = Corpus.from_file("orangecontrib/storynavigation/tests/storynavigator-testdata.tab")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)

    previewer = WidgetPreview(OWSNSettingAnalysis)
    breakpoint()
    previewer.run(set_data=corpus_)
