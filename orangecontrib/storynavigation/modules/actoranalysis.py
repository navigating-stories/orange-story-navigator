"""Modules required for Actor Analysis widget in Story Navigator.
"""

import sys
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from spacy import displacy
import string
import re
from thefuzz import fuzz
from statistics import median

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources

class ActorTagger:
    """Class to perform NLP analysis of actors in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.7/
    """

    def __init__(self, lang):
        self.lang = lang
        self.stopwords = None
        self.__setup_required_nlp_resources(lang)

        self.story_collection = []              # list of story texts that are processed in a session
        self.dataset_level_df_header = []       # column names of dataset (story collection) level dataframe
        self.dataset_level_df = pd.DataFrame()  # complete dataset (story collection) level dataframe
        self.sentence_nlp_models = []           # nlp tagging results for each sentence

        self.html_result = ""                   

        self.agent_prominence_score_max = 0.0
        self.agent_prominence_score_min = 0.0

    def __filter_custom_word_matches(self, story_elements_df, selected_stories, cust_tag_cols):
        cols = []
        words_tagged_with_current_cust_tags_frame = story_elements_df
        story_elements_df[cust_tag_cols] = story_elements_df[cust_tag_cols].astype(str)

        if selected_stories is not None:
            cols = ['segment_id']
            words_tagged_with_current_cust_tags_frame = story_elements_df[story_elements_df['storyid'].isin(selected_stories)]
        else:
            cols = ['storyid', 'segment_id']
        
        combined_df = pd.DataFrame()
        for cust_tag_col in cust_tag_cols:
            
            current_frame = words_tagged_with_current_cust_tags_frame.groupby(cols+[cust_tag_col])[cust_tag_col].agg('count').to_frame("freq").reset_index()
            c_col = [cust_tag_col] * len(current_frame)
            current_frame['classification'] = c_col
            current_frame.rename(columns={cust_tag_col: 'category'}, inplace=True)
            combined_df = pd.concat([combined_df, current_frame], axis=0)

        combined_df = combined_df[combined_df['category'] != 'nan']
        return combined_df.reset_index(drop=True)

    def __filter_rows(self, story_elements_df, pos_tags):
        story_elements_df = story_elements_df.copy()
        story_elements_df['story_navigator_tag'] = story_elements_df['story_navigator_tag'].astype(str)
        story_elements_df['spacy_tag'] = story_elements_df['spacy_tag'].astype(str) 
        matched_df = story_elements_df[story_elements_df['story_navigator_tag'].isin(pos_tags) | story_elements_df['spacy_tag'].isin(pos_tags)]
        matched_df = matched_df.copy()
        
        matched_df['merged_tags'] = np.where(matched_df['story_navigator_tag'] == '-', matched_df['spacy_tag'], matched_df['story_navigator_tag'])
        matched_df['token_start_idx'] = matched_df['token_start_idx'].astype(str)
        matched_df['token_end_idx'] = matched_df['token_end_idx'].astype(str)

        return matched_df
    
    def __do_tagging(self, df):
        ents = []
        if len(df) > 0:
            displacy_tags_list = df['displacy_tag_strings'].tolist()
            for displacy_tag in displacy_tags_list:
                dtag = displacy_tag.split(' | ')
                ents.append({"start": int(float(dtag[0])), "end": int(float(dtag[1])), "label": dtag[2]})

            ents = util.remove_duplicate_tagged_entities(ents) 

        return ents
    
    def __print_html_no_highlighted_tokens(self, sentences):
        html = ''
        for sentence in sentences:
            doc = {"text": sentence, "ents": []}
            options = {"ents": [], "colors": {}}
            html += displacy.render(doc, style="ent", options=options, manual=True)
        return html
    
    def __filter_and_sort_matched_dataframe_by_sentence(self, df, sent, sents):
        order_mapping = {value: index for index, value in enumerate(sents)}
        matched_sent_df = df[df['sentence'] == sent]
        matched_sent_df = matched_sent_df.copy()
        matched_sent_df.loc[:, 'sorting_key'] = matched_sent_df['sentence'].map(lambda value: order_mapping.get(value, len(sents)))
        matched_sent_df_sorted = matched_sent_df.sort_values(by='sorting_key').drop('sorting_key', axis=1)
        return matched_sent_df_sorted
    
    def __do_custom_tagging(self, df, cust_tag_cols):
        df = df.copy()
        df_filtered = df.dropna(subset=cust_tag_cols, how='all')
        multi_custom_tags = []
        ents = []

        for index, row in df_filtered.iterrows():
            current_row_ents = []
            for col in cust_tag_cols:
                if not pd.isna(row[col]) and (row[col] != 'nan'):
                    current_row_ents.append({"start": int(row['token_start_idx']), "end": int(row['token_end_idx']), "label": row[col]})                    

            if len(current_row_ents) > 1:
                concat_labels = '|'.join([d['label'] for d in current_row_ents if 'label' in d and isinstance(d['label'], str)])
                multi_custom_tags.append(concat_labels)
                ents.append({"start": int(current_row_ents[0]['start']), "end": int(current_row_ents[0]['end']), "label": concat_labels})
            elif len(current_row_ents) == 1:
                ents.extend(current_row_ents)

        return ents, multi_custom_tags

    def postag_text(
        self, text, nouns, subjs, custom, selected_prominence_metric, prominence_score_min, story_elements_df
    ):
        """POS-tags story text and returns HTML string which encodes the the tagged text, ready for rendering in the UI

        Args:
            text (string): Story text
            nouns (boolean): whether noun tokens should be tagged
            subjs (boolean): whether subject tokens should be tagged
            custom (boolean): whether custom tags should be highlighted or not
            selected_prominence_metric (float): the selected metric by which to calculate the word prominence score
            prominence_score_min (float): the minimum prominence score for an entity which qualifies it to be tagged
            story_elements_df (pandas.DataFrame): a dataframe with all required nlp tagging information

        Returns:
            string: HTML string representation of POS tagged text
        """
        
        html = ""

        sentences = util.preprocess_text(text)
        
        if story_elements_df is None:
            return self.__print_html_no_highlighted_tokens(sentences)
        
        pos_tags = []
        custom_tag_columns = []
        custom_tags = []
        if nouns:
            pos_tags.append("NSP")
            pos_tags.append("NSNP")
        if subjs:
            pos_tags.append("SP")
            pos_tags.append("SNP")
        if custom:
            custom_tag_columns, custom_tags = util.get_custom_tags_list_and_columns(story_elements_df)

        if len(pos_tags) == 0 and len(custom_tags) == 0:
            return self.__print_html_no_highlighted_tokens(sentences)
        
        matched_df = None

        for sentence in sentences:
            ents = []
            if custom:
                nents = []
                cents = []
                new_color_map = constants.COLOR_MAP
                if nouns or subjs:
                    matched_df = self.__filter_rows(story_elements_df, pos_tags)
                    matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(matched_df, sentence, sentences)
                    matched_sent_df_sorted['displacy_tag_strings'] = matched_sent_df_sorted['token_start_idx'] + ' | ' + matched_sent_df_sorted['token_end_idx'] + ' | ' + matched_sent_df_sorted['merged_tags']
                    nents = self.__do_tagging(matched_sent_df_sorted)
                    matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(story_elements_df, sentence, sentences)
                    cents, multi_tags = self.__do_custom_tagging(matched_sent_df_sorted, custom_tag_columns)
                    for custom_pos_tag in custom_tags + multi_tags:
                        new_color_map[custom_pos_tag] = constants.CUSTOMTAG_HIGHLIGHT_COLOR
                    options = {"ents": pos_tags + custom_tags + multi_tags, "colors": new_color_map}
                else:
                    matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(story_elements_df, sentence, sentences)
                    cents, multi_tags = self.__do_custom_tagging(matched_sent_df_sorted, custom_tag_columns)
                    for custom_pos_tag in custom_tags + multi_tags:
                        new_color_map[custom_pos_tag] = constants.CUSTOMTAG_HIGHLIGHT_COLOR
                    options = {"ents": custom_tags + multi_tags, "colors": new_color_map}
                ents = nents + cents
            else:
                matched_df = self.__filter_rows(story_elements_df, pos_tags)
                matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(matched_df, sentence, sentences)
                matched_sent_df_sorted['displacy_tag_strings'] = matched_sent_df_sorted['token_start_idx'] + ' | ' + matched_sent_df_sorted['token_end_idx'] + ' | ' + matched_sent_df_sorted['merged_tags']
                ents = self.__do_tagging(matched_sent_df_sorted)
                options = {"ents": pos_tags, "colors": constants.COLOR_MAP}

            doc = {"text": sentence, "ents": ents}    
            html += displacy.render(doc, style="ent", options=options, manual=True)

        if custom:
            return util.remove_span_tags_except_custom(html)
        else:
            return util.remove_span_tags(html)

    def __calculate_prominence_score(self, word, selected_prominence_metric):
        """Calculates the promience score for a given word in the story, uses two simple metrics (work in progress and more to follow):
        - Subject frequency : number of times the word appears as a subject of a sentence in the story divided by the number of words in the story
        - Subject frequency (normalized) : number of times the word appears as a subject of a sentence in the story divided by the median subject frequency of a word in the story

        Args:
            word (string): input word
            selected_prominence_metric (string): name of the metric to use

        Returns:
            score: the prominence score of the input word within the story using the specified metric
        """
        score = 0
        # match spacy-tagged token text to the existing dictionary of words in num_occurrences_as_subject
        closest_match_word, successful_match = self.__find_closest_match(
            word, self.num_occurences_as_subject
        )

        if selected_prominence_metric == "Subject frequency (normalized)":
            score = self.num_occurences_as_subject[closest_match_word] / median(
                list(self.num_occurences_as_subject.values())
            )
        elif selected_prominence_metric == "Subject frequency":
            score = (
                self.num_occurences_as_subject[closest_match_word]
                / self.word_count_nostops
            )

        return score

    def __get_max_prominence_score(self):
        """Finds the word in the story with the highest prominence score and returns this score

        Returns:
            highest_score: the score of the word with highest prominence score in the story
        """

        highest_score = 0
        for item in self.word_prominence_scores:
            if self.word_prominence_scores[item] > highest_score:
                highest_score = self.word_prominence_scores[item]
        return highest_score

    def __calculate_agency(self, word):
        """Calculates the agency of a given word (noun) in the story using a custom metric

        Args:
            word (string): input word

        Returns:
            agency_score: the agency score for the input word in the given story
        """
        active_freq = 0
        passive_freq = 0

        for item in self.active_agency_scores:
            active_freq += self.active_agency_scores[item]
        for item in self.passive_agency_scores:
            passive_freq += self.passive_agency_scores[item]

        if active_freq > 0 and passive_freq > 0:
            return (self.active_agency_scores[word] / active_freq) - (
                self.passive_agency_scores[word] / passive_freq
            )
        elif active_freq == 0 and passive_freq > 0:
            return 0 - (self.passive_agency_scores[word] / passive_freq)
        elif active_freq > 0 and passive_freq == 0:
            return self.active_agency_scores[word] / active_freq
        else:
            return 0
    
    def calculate_customfreq_table(self, df, selected_stories=None):
        """Prepares data table for piping to Output variable of widget: frequencies of custom tokens by user

        Args:
            df (pandas dataframe): the dataframe of all categories of custom words by the user

        Returns:
            data table (pandas dataframe)
        """
        cust_tag_cols, cust_tag_names = util.get_custom_tags_list_and_columns(df)
        
        df['token_text_lowercase'] = df['token_text'].str.lower()
        df['token_text_lowercase'] = df['token_text_lowercase'].astype(str)

        if df is None:
            return pd.DataFrame([], columns=constants.CUSTOMFREQ_TABLE_HEADER)

        # rows = []
        # n = 20
        # freq_dict, mapping_dict = self.__filter_custom_word_matches(df, selected_stories=selected_stories, cust_tag_cols=cust_tag_cols)
        return self.__filter_custom_word_matches(df, selected_stories=selected_stories, cust_tag_cols=cust_tag_cols)
        # return 

        # res = dict(
        #     sorted(
        #         freq_dict.items(), key=itemgetter(1), reverse=True
        #     )
        # )

        # words = list(res.keys())

        # for word in words:
        #     # print()
        #     # print('freq: ', freq_dict)
        #     # print()

        #     # print()
        #     # print('map: ', mapping_dict)
        #     # print()
        #     rows.append([word, freq_dict[word], mapping_dict[word][0]])

        # rows.sort(key=lambda x: x[1])
        # return pd.DataFrame(rows[-n:], columns=constants.CUSTOMFREQ_TABLE_HEADER)

    def __prepare_story_elements_frame_for_filtering(self, story_elements_df):
        story_elements_df = story_elements_df.copy()          
        
        strcols = ['token_text', 'sentence', 'associated_action']
        for col in strcols:
            story_elements_df[col] = story_elements_df[col].astype(str)

        story_elements_df['is_sentence_subject_boolean'] = story_elements_df['is_sentence_subject_boolean'].astype(int)
        story_elements_df['active_voice_subject_boolean'] = story_elements_df['active_voice_subject_boolean'].astype(int)
        story_elements_df['storyid'] = 'ST' + story_elements_df['storyid'].astype(str)
        story_elements_df['segment_id'] = 'SE' + story_elements_df['segment_id'].astype(str)

        return story_elements_df
    
    def __update_frame_with_prominence_scores(self, frame, mean_subj_freq_frame):
        frame = frame.copy()
        storyids = frame['storyid'].tolist()

        mean_subj_freq = []
        story_mean_freq_dict = {}
        for storyid in storyids:
            if storyid not in story_mean_freq_dict:
                mean_subj_freq_for_story = mean_subj_freq_frame[mean_subj_freq_frame['storyid'] == storyid]['mean_subj_freq'].tolist()[0]
                mean_subj_freq.append(mean_subj_freq_for_story)
                story_mean_freq_dict[storyid] = mean_subj_freq_for_story
            else:
                mean_subj_freq.append(story_mean_freq_dict[storyid])

        frame['mean_subj_freq_for_story'] = mean_subj_freq
        frame['prominence_sf'] = frame['subj_freq'] / frame['mean_subj_freq_for_story']
        # frame['prominence_sfn'] = frame['subj_freq'] / frame['num_words_in_sentence']
        frame = frame.drop(columns=['mean_subj_freq_for_story'])
        return frame
    
    def generate_actor_analysis_results(self, story_elements_df):
        story_elements_df = self.__prepare_story_elements_frame_for_filtering(story_elements_df)
        result_df = pd.DataFrame()
        
        word_col = next((word for word in story_elements_df.columns if word.startswith('custom_')), None)
        if word_col is None:
            word_col = 'token_text_lowercase'
            
        for word in story_elements_df[story_elements_df['story_navigator_tag'].isin(['SP', 'SNP'])][word_col].unique().tolist(): # loop through unique words that are subjects of sentences in a story
            df_word = story_elements_df[story_elements_df[word_col] == str(word)]
            raw_freq_df = df_word.groupby(['storyid', 'segment_id', word_col])[word_col].agg('count').to_frame("raw_freq").reset_index()
            subj_freq_df = df_word.groupby(['storyid', 'segment_id', word_col])['is_sentence_subject_boolean'].agg('sum').to_frame("subj_freq").reset_index()
            agency_df = df_word.groupby(['storyid', 'segment_id', word_col])['active_voice_subject_boolean'].agg('sum').to_frame("agency").reset_index()        
            mean_subj_freq_df = df_word.groupby(['storyid'])['is_sentence_subject_boolean'].agg('mean').to_frame("mean_subj_freq").reset_index()
            combined_df = pd.merge(raw_freq_df, pd.merge(subj_freq_df, agency_df, on=['storyid', 'segment_id', word_col], how='outer'), on=['storyid', 'segment_id', word_col], how='outer')
            combined_df = self.__update_frame_with_prominence_scores(combined_df, mean_subj_freq_df)
            result_df = pd.concat([result_df, combined_df], axis=0, ignore_index=True)
        
        return result_df
    
    def __setup_required_nlp_resources(self, lang):
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input stories (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
        else:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)

        self.stopwords = [item for item in self.stopwords if len(item) > 0]