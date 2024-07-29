"""Modules required for SettingAnalysis widget in the Orange Story Navigator add-on.
"""

import os
import pandas as pd
import numpy as np
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from collections import Counter
from spacy.matcher import Matcher
from spacy.tokens import Doc


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

    ENTITY_GROUPS = [["EVENT"], ["DATE", "TIME"], ["LOC", "GPE"]]

    def __init__(self, lang, n_segments, text_tuples, callback=None):
        self.text_tuples = text_tuples
        self.n_segments = n_segments
        self.callback = callback

        self.__setup_required_nlp_resources(lang)
        self.nlp = util.load_spacy_pipeline(self.model)

        self.complete_data = self.__process_texts(self.nlp, self.text_tuples, self.callback)
        self.complete_data = self.__select_best_entities(self.complete_data)


    def __setup_required_nlp_resources(self, lang): # TODO: make fct reusable? it's also used in OWSNTagger
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input texts (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
            self.model = constants.NL_SPACY_MODEL
            self.entity_list = constants.NL_ENTITIES_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
        elif lang == constants.EN:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
            self.model = constants.EN_SPACY_MODEL
            self.entity_list =   constants.EN_ENTITIES_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
        else:
            raise ValueError(f"settingsanalysis.py: unknown language {lang}")

        self.stopwords = [item for item in self.stopwords if len(item) > 0]
        self.entity_list = [line.split(",") for line in self.entity_list]


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
        return pd.DataFrame(results, columns=["text", "label", "text id", "character id"]).sort_values(by=["text id", "character id"]).reset_index(drop=True)


    def __analyze_text_with_list(self, text, nlp, entity_list):
        matcher = Matcher(nlp.vocab)
        for entity_group in self.ENTITY_GROUPS:
            patterns = [[{"ORTH": entity_text}]
                for entity_label, entity_text in entity_list
                    if entity_label in entity_group]
            matcher.add(entity_group[0], patterns)
        tokens = nlp(text)
        return { tokens[m[1]].idx: {"text": tokens[m[1]].text,
                                    "label_": nlp.vocab.strings[m[0]]}
                 for m in matcher(tokens) } # presumes entities contain 1 token


    def __process_text(self, text_id, text, nlp):
        """Extract sentences from text and run Spacy analysis on sentences

        Args:
            text_id (int): a number uniquely identifying a specific text
            text (string): text referred to by text_id
            nlp (spacy.language.Language): a spacy language model object to use on the input texts

        Returns:
            list of tuples with entity text, entity label and text id
        """
        spacy_analysis = nlp(text)
        list_analysis = self.__analyze_text_with_list(text, nlp, self.entity_list)
        combined_analysis = { spacy_analysis[entity.start].idx: { "text": entity.text,
                                                                  "label_": entity.label_}
                              for entity in spacy_analysis.ents
                                  if entity.label_ in
                                     ["DATE", "EVENT", "GPE", "LOC", "TIME"] }
        for start in list_analysis:
            combined_analysis[start] = list_analysis[start]
        return [(combined_analysis[start]["text"],
                 combined_analysis[start]["label_"],
                 text_id + 1,
                 start) for start in combined_analysis]


    def __normalize_entities(self, entity_data):
        entity_data_copy = entity_data.copy(deep=True)
        for index, row in entity_data_copy.iterrows():
            for entity_list in self.ENTITY_GROUPS:
                if len(entity_list) > 1 and row["label"] in entity_list[1:]:
                    entity_data_copy.loc[index, "label"] = entity_list[0]
                    break
        return entity_data_copy


    def __select_frequent_entities(self, entity_data):
        counts_series = entity_data[["text", "label", "text id"]].value_counts()
        counts_df = counts_series.reset_index(name="count")
        selected_indices = {}
        for index, row in counts_df.sort_values(by=["count", "text"],
                                                ascending=[False, True]).iterrows():
            key = " ".join([str(row["text id"]), row["label"]])
            if key not in selected_indices.keys():
                selected_indices[key] = [row["text"], row["label"], row["text id"]]
        return list(selected_indices.values())


    def __lookup_selected_values(self, entity_data, selected_values):
        selected_column = len(entity_data) * [""]
        for entity_data_index, row in entity_data.iterrows():
            try:
                selected_values_index = selected_values.index([row["text"],
                                                               row["label"],
                                                               row["text id"]])
                selected_column[entity_data_index] = "selected"
                selected_values.pop(selected_values_index)
            except:
                pass
        return selected_column


    def __select_best_entities(self, entity_data):
        entity_data_copy = self.__normalize_entities(entity_data)
        selected_values = self.__select_frequent_entities(entity_data_copy)
        selected_column = self.__lookup_selected_values(entity_data_copy, 
                                                        selected_values)
        entity_data["selected"] = selected_column
        return entity_data
