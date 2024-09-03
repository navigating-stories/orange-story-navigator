"""Modules required for SettingAnalysis widget in the Orange Story Navigator add-on.
"""

import json
import os
import re
import requests
import time
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

    DATE_LABELS = ["DATE", "TIME"]
    EVENT_LABELS = ["EVENT"]
    LOCATION_LABELS = ["LOC", "FAC", "GPE"]
    ENTITY_GROUPS = [DATE_LABELS, EVENT_LABELS, LOCATION_LABELS]
    ENTITY_LABELS = DATE_LABELS + EVENT_LABELS + LOCATION_LABELS
    ENTITY_CACHE_FILE_NAME = "entity_cache.json"


    def __init__(self, lang, n_segments, text_tuples, callback=None):
        self.text_tuples = text_tuples
        self.n_segments = n_segments
        self.callback = callback

        self.__setup_required_nlp_resources(lang)
        self.nlp = util.load_spacy_pipeline(self.model)

        self.text_analysis = self.__process_texts(self.nlp, self.text_tuples, self.callback)
        self.settings_analysis = self.__select_best_entities(self.text_analysis)


    def __setup_required_nlp_resources(self, lang): # TODO: make fct reusable? it's also used in OWSNTagger
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input texts (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
            self.model = constants.NL_SPACY_MODEL
            self.entity_list = constants.NL_ENTITIES_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
            self.time_words = constants.NL_TIME_WORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
        elif lang == constants.EN:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
            self.model = constants.EN_SPACY_MODEL
            self.entity_list = constants.EN_ENTITIES_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
            self.time_words = constants.EN_TIME_WORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
        else:
            raise ValueError(f"settingsanalysis.py: unknown language {lang}")

        self.stopwords = [item for item in self.stopwords if len(item) > 0]
        self.entity_list = [line.split(",") for line in self.entity_list]


    def __process_texts(self, nlp, text_tuples, callback=None):
        results = []
        for counter, text_tuple in enumerate(text_tuples):
            results.extend(self.__process_text(text_tuple[1], text_tuple[0], nlp))
            if callback:
                callback((100*(counter+1)/len(text_tuples)))
        return pd.DataFrame(results, columns=["text", "label", "text id", "character id", "location type"]).sort_values(by=["text id", "character id"]).reset_index(drop=True)


    def __analyze_text_with_list(self, text, nlp, entity_list):
        matcher = Matcher(nlp.vocab)
        for entity_group in self.ENTITY_GROUPS:
            patterns = [[{"lower": entity_token} for entity_token in entity_text.lower().split()]
                for entity_label, entity_text in entity_list
                    if entity_label in entity_group]
            matcher.add(entity_group[0], patterns)
        tokens = nlp(text)
        return {tokens[m[1]].idx: {
                    "text": " ".join([tokens[token_id].text for token_id in range(m[1], m[2])]),
                    "label_": nlp.vocab.strings[m[0]]
                } for m in matcher(tokens)}


    def __combine_analyses(self, spacy_analysis, list_analysis):
        combined_analysis = { spacy_analysis[entity.start].idx: { "text": entity.text,
                                                                  "label_": entity.label_}
                              for entity in spacy_analysis.ents
                                  if entity.label_ in self.ENTITY_LABELS }
        for start in list_analysis:
            combined_analysis[start] = list_analysis[start]
        return combined_analysis


    def __expand_locations(self, combined_analysis):
        for start in combined_analysis:
            combined_analysis[start]["location type"] = ""
        entities_to_add = {}
        for start in combined_analysis.keys():
            if combined_analysis[start]["label_"] in self.LOCATION_LABELS:
                wikidata_info = self.__get_wikidata_info(combined_analysis[start]["text"])
                if len(wikidata_info) > 0 and "description" in wikidata_info[0]:
                    combined_analysis[start]["location type"] = re.sub("^.* ", "", wikidata_info[0]["description"])
        for start in entities_to_add:
            combined_analysis[start] = entities_to_add[start]
        return combined_analysis


    def __filter_dates(self, combined_analysis):
        to_be_deleted = []
        for start in combined_analysis.keys():
            if combined_analysis[start]["label_"] in self.DATE_LABELS:
                words = combined_analysis[start]["text"].lower().split()
                if len(words) > 0 and words[-1] in self.time_words:
                    to_be_deleted.append(start)
        for start in to_be_deleted:
            del(combined_analysis[start])
        return combined_analysis


    def __process_text(self, text_id, text, nlp):
        spacy_analysis = nlp(text)
        list_analysis = self.__analyze_text_with_list(text, nlp, self.entity_list)
        combined_analysis = self.__combine_analyses(spacy_analysis, list_analysis)
        combined_analysis = self.__expand_locations(combined_analysis)
        combined_analysis = self.__filter_dates(combined_analysis)
        return [(combined_analysis[start]["text"],
                 combined_analysis[start]["label_"],
                 text_id + 1,
                 start,
                 combined_analysis[start]["location type"]) for start in combined_analysis]


    def __normalize_entities(self, entity_data):
        entity_data_copy = entity_data.copy(deep=True)
        for index, row in entity_data_copy.iterrows():
            for entity_list in self.ENTITY_GROUPS:
                if len(entity_list) > 1 and row["label"] in entity_list[1:]:
                    entity_data_copy.loc[index, "label"] = entity_list[0]
                    break
        return entity_data_copy


    def __select_frequent_entities(self, entity_data):
        counts_series = entity_data[["text", "label", "text id", "location type"]].value_counts()
        counts_df = counts_series.reset_index(name="count")
        selected_indices = {}
        for index, row in counts_df.sort_values(by=["count", "text"],
                                                ascending=[False, True]).iterrows():
            key = " ".join([str(row["text id"]), row["label"]])
            if key not in selected_indices.keys() or (
                row["label"] in self.LOCATION_LABELS and
                (len(selected_indices[key][3]) == 0 or not re.search("[A-Z]",selected_indices[key][3][0])) and
                len(row["location type"]) > 0 and re.search("[A-Z]",row["location type"][0])):
                selected_indices[key] = [row["text"], row["label"], row["text id"], row["location type"]]
        return list(selected_indices.values())


    def __select_earliest_entities(self, entity_data):
        counts_series = entity_data[["text", "label", "text id", "location type"]].value_counts()
        counts_df = counts_series.reset_index(name="count").set_index(["text", "label", "text id"])
        selected_indices = {}
        for index, row in entity_data.sort_values(by=["text id", "character id"]).iterrows():
            key = " ".join([str(row["text id"]), row["label"]])
            if (key not in selected_indices.keys() and
                (row["label"] not in self.LOCATION_LABELS or 
                 (re.search("^[A-Z]", row["text"]) and
                  re.search("^[A-Z]", row["location type"])))):
                selected_indices[key] = [row["text"],
                                         row["label"],
                                         row["text id"],
                                         row["location type"],
                                         counts_df.loc[row["text"], row["label"], row["text id"]]["count"]]
        return [list(x)[:4] for x in selected_indices.values()]


    def __lookup_selected_values(self, entity_data, selected_values):
        selected_column = len(entity_data) * [""]
        for entity_data_index, row in entity_data.iterrows():
            try:
                selected_values_index = selected_values.index([row["text"],
                                                               row["label"],
                                                               row["text id"],
                                                               row["location type"]])
                selected_column[entity_data_index] = "selected"
                selected_values.pop(selected_values_index)
            except:
                pass
        return selected_column


    def __select_best_entities(self, entity_data):
        entity_data_copy = self.__normalize_entities(entity_data)
        selected_values = self.__select_earliest_entities(entity_data_copy)
        selected_column = self.__lookup_selected_values(entity_data_copy,
                                                        selected_values)
        entity_data["selected"] = selected_column
        return entity_data


    def __get_wikidata_info(self, entity_name, find_property=False):
        if os.path.isfile(self.ENTITY_CACHE_FILE_NAME):
            with open(self.ENTITY_CACHE_FILE_NAME, "r") as cache_file:
                cache = json.load(cache_file)
            cache_file.close()

        else:
            cache = {}
        if entity_name in cache:
            return cache[entity_name]
        print("__get_wikidata_info: looking up", entity_name)
        url = f"https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "nl",
            "limit": 10,
            "uselang": "nl",
            "search": entity_name
        }
        if find_property:
            params["type"] = "property"
        response = requests.get(url, params=params)
        time.sleep(1)
        data = response.json()
        if 'search' in data.keys():
            cache[entity_name] = data["search"]
        else:
            cache[entity_name] = []
        with open(self.ENTITY_CACHE_FILE_NAME, "w") as cache_file:
            json.dump(cache, cache_file)
        cache_file.close()
        return cache[entity_name]
