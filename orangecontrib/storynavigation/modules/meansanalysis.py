"""Modules required for MeansAnalysis widget in the Orange Story Navigator add-on.
"""

import pandas as pd
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util


class MeansAnalyzer:
    """Class to analyze the means in textual texts
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        language (str): ISO string of the language of the input text
        story_elements (list of lists): tokens with their Spacy analysis
        verb_frames: verb frames indicating means
        means_strategy: strategy to identify means
        callback: function in widget to show the progress of this process
    """


    def __init__(self, language, story_elements, verb_frames, means_strategy, callback=None):
        self.language = language
        self.verb_frames = verb_frames
        self.means_strategy = means_strategy
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        nbr_of_stories = int(story_elements_df.iloc[-1]["storyid"]) + 1

        entities = self.__process_texts(story_elements_df, callback=callback)
        sentence_offsets = self.__compute_sentence_offsets(story_elements_df)
        entities_from_onsets = self.__convert_entities(entities, sentence_offsets)
        self.means_analysis = self.__sort_and_filter_results(entities_from_onsets, nbr_of_stories)


    def __compute_sentence_offsets(self, story_elements_df):
        sentences_df = story_elements_df.groupby(["storyid", "sentence_id"]).first().reset_index()[["storyid", "sentence_id", "sentence"]]
        char_offsets = []
        for index,row in sentences_df.iterrows():
            if int(row["sentence_id"]) == 0:
                char_offset = 0
            else:
                char_offset += len(row["sentence"]) + 1
            char_offsets.append(char_offset)
        sentences_df["char_offset"] = char_offsets
        return sentences_df[["storyid", "sentence_id", "char_offset"]].set_index(["storyid", "sentence_id"])


    def __convert_entities(self, entities, sentences_offsets):
        entities_from_onsets = {}
        for storyid, sentence_id, sentence_data in entities:
            entities_from_onsets[int(storyid)] = {}
            for token_start_id in sentence_data:
                entities_from_onsets[int(storyid)][token_start_id + sentences_offsets.loc[(str(storyid), str(sentence_id))]["char_offset"]] = sentence_data[token_start_id]
        return entities_from_onsets


    def __convert_to_sentences(self, story_elements_df):
        return story_elements_df.groupby(["storyid", "sentence_id"]).agg(lambda x: list(x)).reset_index()


    def __transpose_dict(self, my_dict):
        return [dict(zip(my_dict, col)) for col in zip(*my_dict.values())]


    def __convert_df_to_dict(self, row_sentence_df):
        row_sentence_dict = row_sentence_df.to_dict()
        row_sentence_dict["storyid"] = len(row_sentence_dict["sentence"]) * [row_sentence_dict["storyid"]]
        row_sentence_dict["sentence_id"] = len(row_sentence_dict["sentence"]) * [row_sentence_dict["sentence_id"]]
        return row_sentence_dict


    def __process_texts(self, story_elements_df, callback=None):
        self.sentences_df = self.__convert_to_sentences(story_elements_df)
        entities = []
        for index, row_sentence_df in self.sentences_df.iterrows():
            row_sentence_dict = self.__convert_df_to_dict(row_sentence_df)
            row_sentence_list = self.__transpose_dict(row_sentence_dict)
            row_sentence_transposed = { int(token["token_start_idx"]): token 
                                        for token in row_sentence_list }
            sentence_entities = self.__process_sentence(row_sentence_transposed)
            if sentence_entities:
                entities.append([row_sentence_list[0]["storyid"], 
                                 row_sentence_list[0]["sentence_id"],
                                 sentence_entities])
            if index > 10:
                break
        return entities


    def __matching_dependencies(self, sentence_df, entity_start_id, head_start_id, head_of_head_start_id):

        if sentence_df[head_of_head_start_id]["spacy_tag"] not in ["VERB", "AUX"]:
            return False
        verb_frame_prepositions = [x[1] for x in self.verb_frames] 
        return ((self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAMES and
                 [sentence_df[head_of_head_start_id]["spacy_lemma"],
                  sentence_df[entity_start_id]["spacy_lemma"]] in self.verb_frames) or
                (self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAME_PREPS and
                 sentence_df[entity_start_id]["spacy_lemma"] in verb_frame_prepositions) or
                (self.means_strategy == constants.MEANS_STRATEGY_SPACY_PREPS and
                 sentence_df[entity_start_id]["spacy_tag"] == "ADP"))


    def __expand_means_phrase(self, sentence_df, sentence_entities, char_offset, entity_start_id, head_start_id):
        child_entity_ids = self.__get_head_dependencies(sentence_df, char_offset, entity_start_id, head_start_id)
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
    def __process_sentence(self, sentence_dict, char_offset=0):
        sentence_entities = {}
        for entity_start_id in sorted(sentence_dict.keys()):
            try:
                head_start_id = int(sentence_dict[entity_start_id]["spacy_head_idx"])
                head_of_head_start_id = int(sentence_dict[head_start_id]["spacy_head_idx"])
                if self.language == constants.EN:
                    entity_start_id, head_start_id = head_start_id, entity_start_id
                if self.__matching_dependencies(sentence_dict, entity_start_id, head_start_id, head_of_head_start_id):
                    try:
                        sentence_entities[entity_start_id + char_offset] = { 
                            "label_": "PREP", "text": sentence_dict[entity_start_id]["token_text"] }
                        sentence_entities[head_start_id + char_offset] = { 
                            "label_": "MEANS", "text": sentence_dict[head_start_id]["token_text"] }
                        sentence_entities[head_of_head_start_id + char_offset] = { 
                            "label_": "VERB", "text": sentence_dict[head_of_head_start_id]["token_text"] }
                        self.__expand_means_phrase(sentence_dict, sentence_entities, char_offset, entity_start_id, head_start_id)
                    except Exception as e:
                        #pass
                        print("keyerror1", str(e), sentence_dict[entity_start_id]["sentence"])
            except Exception as e:
                #pass
                print("keyerror2", str(e), sentence_dict[entity_start_id]["sentence"])
        return sentence_entities


    def __get_head_dependencies(self, sentence_df, char_offset, entity_start_id, head_start_id):
        entity_ids = []
        for start_id in sorted(sentence_df.keys()):
            if int(sentence_df[start_id]["spacy_head_idx"]) == head_start_id and start_id != entity_start_id and start_id != head_start_id:
                child_entity_ids = self.__get_head_dependencies(sentence_df, char_offset, entity_start_id, start_id)
                entity_ids.append(start_id)
                entity_ids.extend(child_entity_ids)
        return entity_ids


    def __sort_and_filter_results(self, entities, nbr_of_stories):
        results = []
        index = 0
        for index in range(0, nbr_of_stories):
            if index in entities:
                results.extend([(entities[index][start]["text"],
                                 entities[index][start]["label_"],
                                 index,
                                 int(start)) for start in entities[index]])
            index += 1
        results_df = pd.DataFrame(results, columns=["text", "label", "text id", "character id"]).sort_values(by=["text id", "character id"])
        results_df.insert(3, "storyid", ["ST" + str(int(text_id)) for text_id in results_df["text id"]])
        return results_df[["text", "label", "storyid", "character id"]].reset_index(drop=True)
