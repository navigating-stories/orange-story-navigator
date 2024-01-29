import os
import numpy as np
import scipy.sparse as sp

from Orange.data import Table
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from AnyQt.QtWidgets import QSizePolicy
from Orange.widgets import gui
from Orange.data.pandas_compat import table_from_frame


import spacy
import pandas as pd

import storynavigation.modules.util as util
import storynavigation.modules.constants as constants

from storynavigation.modules.settinganalysis import SettingAnalyzer


class OWSNSettingAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = '3) Setting Analysis'
    description = "Provides tools to help identify the main setting of a story."
    icon = "icons/setting_analysis_icon.png"
    priority = 6480

    NL_SPACY_MODEL = "nl_core_news_sm" 

    class Inputs:
        stories = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        dataset_level_data = Output('Intermediate settings', Table)


    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl' # TODO: add this to widget?
    n_segments = 0 # TODO: add this option to widget this selects the the first entry in the list constants.N_STORY_SEGMENTS 

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories = None 
        # self.prominence_scores = {}
        # self.corpus = None # initialise list of documents (corpus)
        # self.nlp_nl = None # initialise spacy model

        size_policy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.controlArea.setSizePolicy(size_policy)

        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Language",
            value="language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)

        self.select_n_segments_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Number of segments per story",
            value="n_segments",
            items=constants.N_STORY_SEGMENTS,
            sendSelectedValue=True,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )

        self.controlArea.layout().addWidget(self.select_n_segments_combo)

        self.select_language_combo.setEnabled(True)
        self.select_n_segments_combo.setEnabled(True)

        self.compute_data_button = gui.button(
            self.controlArea,
            self,
            label="Analyze setting!",
            callback=self.__analyze_setting,
            width=165,
            height=45,
            toggleButton=False,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            styleSheet="""
                QPushButton {
                    border: 2px solid #555; /* Add border for visibility */
                    border-radius: 10px;
                    padding: 10px;
                }
            """
        )

    


    # def load_spacy_pipeline(self, name): # TODO: can we take this from util?
    #     """Check if the spacy language pipeline was downloaded and load it.
    #     Downloads the language pipeline if not available.
    
    #     Args:
    #         name (string): Name of the spacy language.
    
    #     Returns:
    #         spacy.language.Language: The spacy language pipeline
    #     """
    #     if spacy.util.is_package(name):
    #         nlp = spacy.load(name)
    #     else:
    #         os.system(f"spacy download {name}")
    #         nlp = spacy.load(name)
    #     return nlp

    @Inputs.stories
    def set_stories(self, stories=None):
        idx = 0 # TODO: replace this when #32 is merged
        self.stories = []
        for document in stories:
            text = ''
            for field in stories.domain.metas:
                text_field_name = str(field)
                if text_field_name.lower() in ['text', 'content']:
                    text = str(stories[idx, text_field_name])

            if len(text) > 0:
                self.stories.append((text, idx))

            idx += 1

        # self.nlp_nl = self.load_spacy_pipeline(self.NL_SPACY_MODEL)
        # self.nlp_nl.add_pipe("merge_noun_chunks")
        # self.corpus = corpus
        # # self._generate_network(self.corpus)
        # self.get_embeddings()

    def reset_widget(self):
        self.corpus = None
        self.Warning.clear()


    def __analyze_setting(self):
        """
        Processes the corpus and extracts embeddings for selected words.
        """
        n_segments = int(self.n_segments)
        if n_segments == 0: # if the user does not choose explicitly the value in the menu, the value will be 0.
            n_segments = 1 
        analyzer = SettingAnalyzer(
             lang=self.language, n_segments=n_segments,
             text_tuples=self.stories
        )
        self.Outputs.dataset_level_data.send(table_from_frame(analyzer.complete_data))



if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    corpus_ = Corpus.from_file("orangecontrib/storynavigation/tests/storynavigator-testdata.tab")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)

    previewer = WidgetPreview(OWSNSettingAnalysis)
    breakpoint()
    previewer.run(set_stories=corpus_)
