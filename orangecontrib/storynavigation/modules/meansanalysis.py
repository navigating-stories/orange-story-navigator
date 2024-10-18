"""Modules required for MeansAnalysis widget in the Orange Story Navigator add-on.
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
from pathlib import Path


class MeansAnalyzer:
    """Class to analyze the means in textual texts
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        language (str): ISO string of the language of the input text
        n_segments (int): Number of segments to split each text into
        text_tuples (list): binary tuple: text (str) and storyid
        story_elements (list of lists): tokens with their Spacy analysis
        callback: function in widget to show the progress of this process
    """

    DATE_LABELS = ["DATE", "TIME"]
    EVENT_LABELS = ["EVENT"]
    LOCATION_LABELS = ["LOC", "FAC", "GPE"]
    ENTITY_GROUPS = [DATE_LABELS, EVENT_LABELS, LOCATION_LABELS]
    ENTITY_LABELS = ['MEANS', 'PREP', 'VERB'] + DATE_LABELS + EVENT_LABELS + LOCATION_LABELS


    def __init__(self, language, n_segments, text_tuples, story_elements, verb_frames, means_strategy, callback=None):
        self.language = language
        self.text_tuples = text_tuples
        self.n_segments = n_segments
        self.verb_frames = verb_frames
        self.means_strategy = means_strategy
        self.callback = callback

        self.__setup_required_nlp_resources(language)
        self.nlp = util.load_spacy_pipeline(self.model)

        entities = self.__process_texts(story_elements)
        self.text_analysis = self.__process_texts_with_entities(self.nlp, self.text_tuples, entities, self.callback)
        self.means_analysis = self.__select_best_entities(self.text_analysis)


    def __process_texts(self, story_elements):
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        last_story_id = -1
        last_sentence = ""
        char_offset = 0
        entities = []
        sentence_df = {}
        for index, row in story_elements_df.iterrows():
            story_id = row["storyid"]
            if story_id != last_story_id:
                if len(sentence_df) > 0:
                   sentence_entities = self.process_sentence(sentence_df, char_offset)
                   for key in sentence_entities.keys():
                       entities[-1][key] = sentence_entities[key]
                sentence_df = {}
                entities.append({})
                last_entity_class = "O"
                last_entity_start_id = -1
                last_sentence = ""
                char_offset = 0
            sentence = row["sentence"]
            if sentence != last_sentence:
                if len(sentence_df) > 0:
                   sentence_entities = self.process_sentence(sentence_df, char_offset)
                   for key in sentence_entities.keys():
                       entities[-1][key] = sentence_entities[key]
                sentence_df = {}
                if len(last_sentence) > 0:
                    char_offset += 1 + len(last_sentence)
                last_sentence = sentence
            entity_start_id = int(row["token_start_idx"])
            sentence_df[entity_start_id] = row
            last_story_id = story_id
        return entities


    def matching_dependencies(self, sentence_df, entity_start_id, head_start_id, head_of_head_start_id):

        if sentence_df[head_of_head_start_id]["spacy_tag"] not in ["VERB", "AUX"]:
            return False
        verb_frame_prepositions = [x[1] for x in self.verb_frames] 
        return ((self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAMES and
                 [sentence_df[head_of_head_start_id]["spacy_lemma"],
                  sentence_df[entity_start_id]["token_text"]] in self.verb_frames) or
                (self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAME_PREPS and
                 sentence_df[entity_start_id]["token_text"] in verb_frame_prepositions) or
                (self.means_strategy == constants.MEANS_STRATEGY_SPACY_PREPS and
                 sentence_df[entity_start_id]["spacy_tag"] == "ADP"))


    def expand_means_phrase(self, sentence_df, sentence_entities, char_offset, entity_start_id, head_start_id):
        child_entity_ids = self.get_head_dependencies(sentence_df, char_offset, entity_start_id, head_start_id)
        processed_ids = []
        for child_entity_id in sorted(child_entity_ids, reverse=True):
            child_entity_text = sentence_df[child_entity_id]["token_text"]
            entity_gap_size = head_start_id - len(child_entity_text) - child_entity_id
            if child_entity_id not in processed_ids and entity_gap_size in [1, 2]:
                in_between_text = " " if entity_gap_size == 1 else ", "
                sentence_entities[child_entity_id + char_offset] = {
                   "text": sentence_df[child_entity_id]["token_text"] + in_between_text + sentence_entities[head_start_id + char_offset]["text"],
                   "label_": "MEANS" }
                del(sentence_entities[head_start_id + char_offset])
                head_start_id = child_entity_id
                processed_ids.append(child_entity_id)
        for child_entity_id in sorted(child_entity_ids):
            child_entity_text = sentence_df[child_entity_id]["token_text"]
            entity_gap_size = child_entity_id - head_start_id - len(sentence_entities[head_start_id + char_offset]["text"])
            if child_entity_id not in processed_ids and entity_gap_size in [1, 2]:
                in_between_text = " " if entity_gap_size == 1 else ", "
                sentence_entities[head_start_id + char_offset]["text"] += in_between_text + sentence_df[child_entity_id]["token_text"]
                processed_ids.append(child_entity_id)
            if child_entity_id not in processed_ids:
                print(sentence_df[entity_start_id]["token_text"], sentence_df[head_start_id]["token_text"], "skipping means word", sentence_df[child_entity_id]["token_text"])


    # nl head relations: PREP -> MEANS -> VERB
    # en head relations: MEANS -> PREP -> VERB
    def process_sentence(self, sentence_df, char_offset):
        sentence_entities = {}
        for entity_start_id in sorted(sentence_df.keys()):
            try:
                head_start_id = int(sentence_df[entity_start_id]["spacy_head_idx"])
                head_of_head_start_id = int(sentence_df[head_start_id]["spacy_head_idx"])
                if self.language == constants.EN:
                    entity_start_id, head_start_id = head_start_id, entity_start_id
                if self.matching_dependencies(sentence_df, entity_start_id, head_start_id, head_of_head_start_id):
                    try:
                        sentence_entities[entity_start_id + char_offset] = { 
                            "label_": "PREP", "text": sentence_df[entity_start_id]["token_text"] }
                        sentence_entities[head_start_id + char_offset] = { 
                            "label_": "MEANS", "text": sentence_df[head_start_id]["token_text"] }
                        sentence_entities[head_of_head_start_id + char_offset] = { 
                            "label_": "VERB", "text": sentence_df[head_of_head_start_id]["token_text"] }
                        self.expand_means_phrase(sentence_df, sentence_entities, char_offset, entity_start_id, head_start_id)
                    except Exception as e:
                        #pass
                        print("keyerror1", str(e), sentence_df[entity_start_id]["sentence"])
            except Exception as e:
                #pass
                print("keyerror2", str(e), sentence_df[entity_start_id]["sentence"])
        return sentence_entities


    def process_sentence_en(self, sentence_df, char_offset):
        sentence_entities = {}
        for entity_start_id in sorted(sentence_df.keys()):
            try:
                head_start_id = int(sentence_df[entity_start_id]["spacy_head_idx"])
                head_of_head_start_id = int(sentence_df[head_start_id]["spacy_head_idx"])
                if self.matching_dependencies(sentence_df, head_start_id, entity_start_id, head_of_head_start_id):
                    try:
                        sentence_entities[entity_start_id + char_offset] = {
                            "label_": "MEANS", "text": sentence_df[entity_start_id]["token_text"] }
                        sentence_entities[head_start_id + char_offset] = {
                            "label_": "PREP", "text": sentence_df[head_start_id]["token_text"] }
                        sentence_entities[head_of_head_start_id + char_offset] = {
                            "label_": "VERB", "text": sentence_df[head_of_head_start_id]["token_text"] }
                        self.expand_means_phrase(sentence_df, sentence_entities, char_offset, head_start_id, entity_start_id)
                    except Exception as e:
                        #pass
                        print("keyerror1", str(e), sentence_df[entity_start_id]["sentence"])
            except Exception as e:
                #pass
                print("keyerror2", str(e), sentence_df[entity_start_id]["sentence"])
        return sentence_entities


    def get_head_dependencies(self, sentence_df, char_offset, entity_start_id, head_start_id):
        entity_ids = []
        for start_id in sorted(sentence_df.keys()):
            if int(sentence_df[start_id]["spacy_head_idx"]) == head_start_id and start_id != entity_start_id and start_id != head_start_id:
                child_entity_ids = self.get_head_dependencies(sentence_df, char_offset, entity_start_id, start_id)
                entity_ids.append(start_id)
                entity_ids.extend(child_entity_ids)
        return entity_ids


    def __setup_required_nlp_resources(self, language):
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            language (string): the ISO code for the language of the input texts (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if language == constants.NL:
            self.model = constants.NL_SPACY_MODEL
        elif language == constants.EN:
            self.model = constants.EN_SPACY_MODEL
        else:
            raise ValueError(f"meansanalysis.py: unknown language {language}")


    def __sort_and_filter_results(self, results):
        results = [(x[0], x[1], int(x[2]), x[3], x[4]) for x in results]
        results_df = pd.DataFrame(results, columns=["text", "label", "text id", "character id", "location type"]).sort_values(by=["text id", "character id"])
        results_df.insert(3, "storyid", ["ST" + str(text_id) for text_id in results_df["text id"]])
        return results_df[["text", "label", "storyid", "character id", "location type"]].reset_index(drop=True)


    def __process_texts_with_entities(self, nlp, text_tuples, entities, callback=None):
        results = []
        index = 0
        for entities_per_text in entities:
            results.extend(self.__process_text(text_tuples[index][1], text_tuples[index][0], nlp, entities_per_text))
            if callback:
                callback((100*(index+1)/len(entities)))
            index += 1
        return self.__sort_and_filter_results(results)


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
                if len(words) > 0 and words[-1] in self.verb_frames:
                    to_be_deleted.append(start)
        for start in to_be_deleted:
            del(combined_analysis[start])
        return combined_analysis


    def __process_text(self, text_id, text, nlp, spacy_analysis):
        combined_analysis = spacy_analysis
        combined_analysis = self.__expand_locations(combined_analysis)
        combined_analysis = self.__filter_dates(combined_analysis)
        return [(combined_analysis[start]["text"],
                 combined_analysis[start]["label_"],
                 text_id,
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


    def __select_earliest_entities(self, entity_data):
        counts_series = entity_data[["text", "label", "storyid", "location type"]].value_counts()
        counts_df = counts_series.reset_index(name="count").set_index(["text", "label", "storyid"])
        selected_indices = {}
        for index, row in entity_data.iterrows():
            key = " ".join([str(row["storyid"]), row["label"]])
            if (key not in selected_indices.keys() and
                (row["label"] not in self.LOCATION_LABELS or 
                 (re.search("^[A-Z]", row["text"]) and
                  re.search("^[A-Z]", row["location type"])))):
                selected_indices[key] = [row["text"],
                                         row["label"],
                                         row["storyid"],
                                         row["location type"],
                                         counts_df.loc[row["text"], row["label"], row["storyid"]]["count"]]
        return [list(x)[:4] for x in selected_indices.values()]


    def __lookup_selected_values(self, entity_data, selected_values):
        selected_column = len(entity_data) * [""]
        for entity_data_index, row in entity_data.iterrows():
            try:
                selected_values_index = selected_values.index([row["text"],
                                                               row["label"],
                                                               row["storyid"],
                                                               row["location type"]])
                selected_column[entity_data_index] = "selected"
                selected_values.pop(selected_values_index)
            except Exception as e:
                pass
                #print("__lookup_selected_values: error", str(e))
        return selected_column


    def __select_best_entities(self, entity_data):
        entity_data_copy = self.__normalize_entities(entity_data)
        selected_values = self.__select_earliest_entities(entity_data_copy)
        selected_column = self.__lookup_selected_values(entity_data_copy,
                                                        selected_values)
        entity_data["selected"] = selected_column
        return entity_data


    def __get_wikidata_info(self, entity_name, find_property=False):
        if self.ENTITY_CACHE_FILE_NAME and os.path.isfile(self.ENTITY_CACHE_FILE_NAME):
            with open(self.ENTITY_CACHE_FILE_NAME, "r") as cache_file:
                cache = json.load(cache_file)
            cache_file.close()
        else:
            cache = {}
        if entity_name in cache and len(cache[entity_name]) > 0:
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
        try:
            response = requests.get(url, params=params)
            time.sleep(1)
            data = response.json()
        except Exception as e:
            print("lookup failure: no network connection?", str(e))
            data = {}
        if 'search' in data.keys() and len(data["search"]) > 0:
            cache[entity_name] = data["search"]
        else:
            cache[entity_name] = [{"not found": True}]
        if self.ENTITY_CACHE_FILE_NAME:
            try:
                with open(self.ENTITY_CACHE_FILE_NAME, "w") as cache_file:
                    json.dump(cache, cache_file)
                cache_file.close()
            except Exception as e:
                print("__get_wikidata_info: error:", str(e))
        return cache[entity_name]
