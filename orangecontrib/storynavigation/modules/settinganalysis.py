"""Modules required for SettingAnalysis widget in the Orange Story Navigator add-on.
"""

import os
import pandas as pd
import numpy as np
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from collections import Counter



class SettingAnalyzer:
    """Class to analyze the setting in textual texts
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        lang (str): ISO string of the language of the input text
        n_segments (int): Number of segments to split each text into
        text_tuples (list): binary tuple: text (str) and text id
        callback: function in widget to show the progress of this process
    """


    def __init__(self, lang, n_segments, text_tuples, callback=None):
        self.text_tuples = text_tuples
        self.n_segments = n_segments
        self.callback = callback

        self.__setup_required_nlp_resources(lang)
        self.nlp = util.load_spacy_pipeline(self.model)

        self.complete_data = self.__process_texts(self.nlp, self.text_tuples, self.callback)


    def __setup_required_nlp_resources(self, lang): # TODO: make fct reusable? it's also used in OWSNTagger
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input texts (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.model = constants.NL_SPACY_MODEL
        elif lang == constants.EN:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.model = constants.EN_SPACY_MODEL
        else:
            raise ValueError(f"settingsanalysis.py: unknown language {lang}")

        self.stopwords = [item for item in self.stopwords if len(item) > 0]


    def __process_texts(self, nlp, text_tuples, callback=None):
        """Run NLP model, lemmatize tokens and collect them in a dataframe (one row per unique lemma).

        Args:
            nlp (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sentences'
            text_tuples (list): each element of the list is a binary tuple. The first component is the text of the text (string) and
            the second component is a number (int) uniquely identifying that text in the given list

        Returns:
            pandas.DataFrame: a dataframe containing all tagging data for all texts in the given list
        """
        results = []
        for counter, text_tuple in enumerate(text_tuples):
            results.extend(self.__process_text(text_tuple[1], text_tuple[0], nlp))
            if callback:
                callback((100*(counter+1)/len(text_tuples)))
        return pd.DataFrame(results, columns=["text", "label", "text id"])


    def __analyze_sentences(self, sentences, nlp):
        return [ nlp(sentence) for sentence in sentences if (len(sentence.split()) > 0) ]


    def __process_text(self, text_id, text, nlp):
        """Extract sentences from text and run Spacy analysis on sentences

        Args:
            text_id (int): a number uniquely identifying a specific text
            text (string): text referred to by text_id
            nlp (spacy.language.Language): a spacy language model object to use on the input texts

        Returns:
            list of tuples with entity text, entity label and text id
        """
        sentences = util.preprocess_text(text)
        analyzed_sentences = self.__analyze_sentences(sentences, nlp)
        return [(entity.text, entity.label_, 1 + int(text_id))
                    for analyzed_sentence in analyzed_sentences 
                        for entity in analyzed_sentence.ents 
                            if entity.label_ in 
                                ["DATE", "EVENT", "GPE", "LOC", "TIME"] ]

