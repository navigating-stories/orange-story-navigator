"""Modules required for Action Analysis widget in Story Navigator.
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
from nltk.tokenize import RegexpTokenizer
import json

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources


class ActionTagger:
    """Class to perform NLP analysis of actors in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.7/
    """

    def __init__(self, lang):
        self.lang = lang
        self.stopwords = None
        self.__setup_required_nlp_resources(lang)
        self.html_result = ""
        self.num_sents_in_stories = {}
        self.tagging_cache = {}

        # self.stopwords = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
        # self.stopwords = [item for item in self.stopwords if len(item) > 0]
        # self.pronouns = self.NL_PRONOUNS_FILE.read_text(encoding="utf-8").split(os.linesep)
        # self.pronouns = [item for item in self.pronouns if len(item) > 0]

        # self.past_tense_verbs = self.NL_PAST_TENSE_FILE.read_text(encoding="utf-8").split(os.linesep)
        # self.past_tense_verbs = [item for item in self.past_tense_verbs if len(item) > 0]
        # self.present_tense_verbs = self.NL_PRESENT_TENSE_FILE.read_text(encoding="utf-8").split(os.linesep)
        # self.present_tense_verbs = [item for item in self.present_tense_verbs if len(item) > 0]
        # self.false_positive_verbs = self.NL_FALSE_POSITIVE_VERB_FILE.read_text(encoding="utf-8").split(os.linesep)
        # self.false_positive_verbs = [item for item in self.false_positive_verbs if len(item) > 0]

        # self.html_result = ""

        # # Other counts initialisation
        # self.word_count = 0
        # self.word_count_nostops = 0
        # self.sentence_count = 0
        # self.sentence_count_per_word = {}
        # self.active_agency_scores = {}
        # self.passive_agency_scores = {}
        # self.past_verb_count = {}
        # self.present_verb_count = {}
        # self.num_occurences = {}
        # self.num_occurences_as_subject = {}
        # self.noun_action_dict = {}

        # self.nlp = util.load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        # self.agent_prominence_score_max = 0.0
        # self.agent_prominence_score_min = 0.0

        # # Index of word prominence scores for each word in story
        # self.word_prominence_scores = {}
        # self.sentence_nlp_models = []

        # # POS counts initialisation
        # self.noun_count = 0
        # self.verb_count = 0
        # self.adjective_count = 0

    def __update_postagging_metrics(self, tagtext, token):
        """After pos-tagging a particular token, this method is executed to calculate the noun-action dictionary

        Args:
            tagtext (string): the string representation of the input token from the story text

        """

        vb = util.find_verb_ancestor(token)
        if vb is not None:
            if tagtext in self.noun_action_dict:
                self.noun_action_dict[tagtext].append(vb.text)
            else:
                self.noun_action_dict[tagtext] = []

    def __calculate_pretagging_metrics(self, sentences):
        """Before pos-tagging commences, this method is executed to calculate some basic story metrics
        including word count (with and without stopwords) and sentence count.

        Args:
            sentences (list): list of string sentences from the story
        """

        self.sentence_count = len(sentences)
        for sentence in sentences:
            words = sentence.split()
            tokens = []
            for word in words:
                if len(word) > 1:
                    if word[len(word) - 1] in string.punctuation:
                        tokens.append(word[: len(word) - 1].lower().strip())
                    else:
                        tokens.append(word.lower().strip())

            self.word_count += len(tokens)

            if len(self.stopwords) > 0:
                for token in tokens:
                    if token not in self.stopwords:
                        self.word_count_nostops += 1
            else:
                self.word_count_nostops = self.word_count

    def __print_html_no_highlighted_tokens(self, sentences):
        html = ''
        for sentence in sentences:
            doc = {"text": sentence, "ents": []}
            options = {"ents": [], "colors": {}}
            html += displacy.render(doc, style="ent", options=options, manual=True)
        return html
    
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
    
    def __filter_and_sort_matched_dataframe_by_sentence(self, df, sent, sents):
        order_mapping = {value: index for index, value in enumerate(sents)}
        matched_sent_df = df[df['sentence'] == sent]
        matched_sent_df = matched_sent_df.copy()
        matched_sent_df.loc[:, 'sorting_key'] = matched_sent_df['sentence'].map(lambda value: order_mapping.get(value, len(sents)))
        matched_sent_df_sorted = matched_sent_df.sort_values(by='sorting_key').drop('sorting_key', axis=1)
        return matched_sent_df_sorted

    def __do_tagging(self, df):
        ents = []
        if len(df) > 0:
            displacy_tags_list = df['displacy_tag_strings'].tolist()
            for displacy_tag in displacy_tags_list:
                dtag = displacy_tag.split(' | ')
                ents.append({"start": int(float(dtag[0])), "end": int(float(dtag[1])), "label": dtag[2]})

            ents = util.remove_duplicate_tagged_entities(ents) 

        return ents

    def __do_custom_tagging(self, df, cust_tag_cols):
        df = df.copy()

        df_filtered = df.dropna(subset=cust_tag_cols, how='all')
        multi_custom_tags = []
        ents = []

        for index, row in df_filtered.iterrows():
            current_row_ents = []
            for col in cust_tag_cols:
                if (not pd.isna(row[col])) and (row[col] != 'nan'):
                    current_row_ents.append({"start": int(row['token_start_idx']), "end": int(row['token_end_idx']), "label": row[col]})                    

            # Convert each dictionary to a tuple of (key, value) pairs
            tuple_list = [tuple(sorted(d.items())) for d in current_row_ents]
            # Use a set to remove duplicates
            unique_tuples = set(tuple_list)
            # Convert back to dictionaries
            current_row_ents = [dict(t) for t in unique_tuples]

            if len(current_row_ents) > 1:
                concat_labels = '|'.join([d['label'] for d in current_row_ents if 'label' in d and isinstance(d['label'], str)])
                multi_custom_tags.append(concat_labels)
                ents.append({"start": int(current_row_ents[0]['start']), "end": int(current_row_ents[0]['end']), "label": concat_labels})
            elif len(current_row_ents) == 1:
                ents.extend(current_row_ents)

        return ents, multi_custom_tags
    
    def __postag_sents(
            self, sentences, past_vbz, present_vbz, custom, story_elements_df
    ):
        html = ""
        
        if story_elements_df is None:
            return self.__print_html_no_highlighted_tokens(sentences)

        pos_tags = []
        custom_tag_columns = []
        custom_tags = []
        if past_vbz:
            pos_tags.append("PAST_VB")
        if present_vbz:
            pos_tags.append("PRES_VB")
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
                if past_vbz or present_vbz:
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

            # Convert each dictionary to a tuple of (key, value) pairs
            tuple_list = [tuple(sorted(d.items())) for d in ents]
            # Use a set to remove duplicates
            unique_tuples = set(tuple_list)
            # Convert back to dictionaries
            ents = [dict(t) for t in unique_tuples]

            doc = {"text": sentence, "ents": ents}    
            html += displacy.render(doc, style="ent", options=options, manual=True)

        if custom:
            self.html_result = util.remove_span_tags_except_custom(html)
            return self.html_result
        else:
            self.html_result = util.remove_span_tags(html)
            return self.html_result

    def postag_text(
            self, text, past_vbz, present_vbz, custom, story_elements_df
    ):
        """POS-tags story text and returns HTML string which encodes the the tagged text, ready for rendering in the UI

        Args:
            text (string): Story text
            nouns (boolean): whether noun tokens should be tagged
            subjs (boolean): whether subject tokens should be tagged
            selected_prominence_metric: the selected metric by which to calculate the word prominence score

        Returns:
            string: HTML string representation of POS tagged text
        """
        
        if story_elements_df is None or len(story_elements_df) == 0:
            sentences = util.preprocess_text(text)
            return self.__print_html_no_highlighted_tokens(sentences)
        
        sorted_df = story_elements_df.sort_values(by=['storyid', 'sentence_id'], ascending=True)
        sentences = sorted_df['sentence'].unique().tolist()

        selected_storyid = story_elements_df['storyid'].unique().tolist()[0]
        specific_tag_choice_html = (str(int(past_vbz)) + str(int(present_vbz)) + str(int(custom)))
        if selected_storyid in self.tagging_cache:
            if specific_tag_choice_html in self.tagging_cache[str(selected_storyid)]: # tagging info already generated, just lookup cached results
                return self.tagging_cache[str(selected_storyid)][specific_tag_choice_html]
            else:
                self.tagging_cache[str(selected_storyid)][specific_tag_choice_html] = self.__postag_sents(sentences, past_vbz, present_vbz, custom, story_elements_df)
        else:
            self.tagging_cache[str(selected_storyid)] = {}
            self.tagging_cache[str(selected_storyid)][specific_tag_choice_html] = self.__postag_sents(sentences, past_vbz, present_vbz, custom, story_elements_df)

        return self.tagging_cache[str(selected_storyid)][specific_tag_choice_html]
    
        # self.__calculate_pretagging_metrics(sentences)

        # # pos tags that the user wants to highlight
        # pos_tags = []

        # # add pos tags to highlight according to whether the user has selected them or not
        # if past_vbz:
        #     pos_tags.append("PAST_VB")
        # if present_vbz:
        #     pos_tags.append("PRES_VB")

        # # output of this function
        # html = ""

        # # generate and store nlp tagged models for each sentence
        # if self.sentence_nlp_models is None or len(self.sentence_nlp_models) == 0:
        #     # sentence_nlp_models = []
        #     for sentence in sentences:
        #         tagged_sentence = self.nlp(sentence.replace("`", "").replace("'", "").replace("‘", "").replace("’", ""))
        #         self.sentence_nlp_models.append(tagged_sentence)

        #     self.__calculate_action_type_count(self.sentence_nlp_models)

        # # loop through model to filter out those words that need to be tagged (based on user selection and prominence score)
        # for sentence, tagged_sentence in zip(sentences, self.sentence_nlp_models):
        #     tags = []
        #     tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")
        #     spans = list(tokenizer.span_tokenize(sentence))

        #     for token in tagged_sentence:
        #         tags.append((token.text, token.pos_, token.tag_, token.dep_, token))

        #     ents = []
        #     for tag, span in zip(tags, spans):
        #         normalised_token, is_valid_token = self.__is_valid_token(tag)
        #         if is_valid_token:
        #             if ((tag[4].text.lower().strip() in self.past_tense_verbs) or (tag[4].text.lower().strip()[:2] == "ge")) and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # past tense
        #                 ents.append(
        #                     {"start": span[0], "end": span[1], "label": "PAST_VB"}
        #                 )
        #             else:
        #                 if (tag[4].pos_ == "VERB") and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # present tense
        #                     ents.append(
        #                         {"start": span[0], "end": span[1], "label": "PRES_VB"}
        #                     )
                        
        #                 elif tag[4].pos_ in ["NOUN", "PRON", "PROPN"]: # non-verbs (for noun-action table)
        #                     self.__update_postagging_metrics(
        #                         tag[4].text.lower().strip(), tag[4]
        #                     )

        #     # specify sentences and filtered entities to tag / highlight
        #     doc = {"text": sentence, "ents": ents}

        #     # specify colors for highlighting each entity type
        #     colors = {}
        #     if past_vbz:
        #         colors["PAST_VB"] = constants.ACTION_PAST_HIGHLIGHT_COLOR
        #     if present_vbz:
        #         colors["PRES_VB"] = constants.ACTION_PRESENT_HIGHLIGHT_COLOR

        #     # collect the above config params together
        #     options = {"ents": pos_tags, "colors": colors}
        #     # give all the params to displacy to generate HTML code of the text with highlighted tags
        #     html += displacy.render(doc, style="ent", options=options, manual=True)

        # self.html_result = html
        # # return html
        # return util.remove_span_tags(html)
    
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
    
    def calculate_customfreq_table(self, df, selected_stories=None):
        """Prepares data table for piping to Output variable of widget: frequencies of custom tokens by user

        Args:
            df (pandas dataframe): the dataframe of all categories of custom words by the user

        Returns:
            data table (pandas dataframe)
        """
        cust_tag_cols, cust_tag_names = util.get_custom_tags_list_and_columns(df)

        df['token_text'] = df['token_text'].astype(str)
        df['token_text_lowercase'] = df['token_text'].str.lower()

        if df is None:
            return pd.DataFrame([], columns=constants.CUSTOMFREQ_TABLE_HEADER)

        return self.__filter_custom_word_matches(df, selected_stories=selected_stories, cust_tag_cols=cust_tag_cols)

    def __is_valid_token(self, token):
        """Verifies if token is valid word

        Args:
            token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

        Returns:
            string, boolean : cleaned token text, True if the input token is a valid word, False otherwise
        """

        word = util.get_normalized_token(token)
        return word, (word not in self.stopwords) and len(word) > 1

    def __calculate_action_type_count(self, sent_models):
        """Calculates the frequency of mentions for each word in the story:

        Args:
            sents (list): list of all sentences (strings) from the input story
            sent_models (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sents'
        """

        for sent_model in sent_models:
            for token in sent_model:
                normalised_token, is_valid_token = self.__is_valid_token(token)
                if is_valid_token:
                    if ((token.text.lower().strip() in self.past_tense_verbs) or (token.text.lower().strip()[:2] == "ge")) and (token.text.lower().strip() not in self.false_positive_verbs):  # past tense
                    # if token.pos_ == "VERB":
                    #     vb_tense = token.morph.get("Tense")
                    #     if vb_tense == "Past":
                        if token.text.lower().strip() in self.past_verb_count:
                            self.past_verb_count[token.text.lower().strip()] += 1
                        else:
                            self.past_verb_count[token.text.lower().strip()] = 1
                    else:
                        if token.pos_ == "VERB" and (token.text.lower().strip() not in self.false_positive_verbs):
                    # elif vb_tense == "Pres":
                            if token.text.lower().strip() in self.present_verb_count:
                                self.present_verb_count[token.text.lower().strip()] += 1
                            else:
                                self.present_verb_count[token.text.lower().strip()] = 1
                        # else:
                        #     if token.text.lower().strip()[:2] == "ge":  # past tense
                        #         if token.text.lower().strip() in self.past_verb_count:
                        #             self.past_verb_count[
                        #                 token.text.lower().strip()
                        #             ] += 1
                        #         else:
                        #             self.past_verb_count[token.text.lower().strip()] = 1
                        #     else:
                        #         if (
                        #             token.text.lower().strip()
                        #             in self.present_verb_count
                        #         ):
                        #             self.present_verb_count[
                        #                 token.text.lower().strip()
                        #             ] += 1
                        #         else:
                        #             self.present_verb_count[
                        #                 token.text.lower().strip()
                        #             ] = 1

    def calculate_metrics_freq_table(self):
        """Prepares data table for piping to Output variable of widget: frequency of verbs in story

        Returns:
            data table (pandas dataframe)
        """

        rows = []
        n = 20

        res_past = dict(
            sorted(self.past_verb_count.items(), key=itemgetter(1), reverse=True)
        )
        past_verbs = list(res_past.keys())
        res_pres = dict(
            sorted(self.present_verb_count.items(), key=itemgetter(1), reverse=True)
        )
        pres_verbs = list(res_pres.keys())

        for past_verb in past_verbs:
            rows.append([past_verb, self.past_verb_count[past_verb]])

        for pres_verb in pres_verbs:
            rows.append([pres_verb, self.present_verb_count[pres_verb]])

        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.ACTION_FREQ_TABLE_HEADER)

    def calculate_metrics_tensefreq_table(self):
        """Prepares data table for piping to Output variable of widget: number of verbs in each tense category

        Returns:
            data table (pandas dataframe)
        """
        rows = []

        past_total = 0
        pres_total = 0
        for past_verb in self.past_verb_count:
            past_total += self.past_verb_count[past_verb]

        for pres_verb in self.present_verb_count:
            pres_total += self.present_verb_count[pres_verb]

        rows.append(["Past tense", past_total])
        rows.append(["Present tense", pres_total])

        return pd.DataFrame(rows, columns=constants.ACTION_TENSEFREQ_TABLE_HEADER)

    def generate_noun_action_table(self):
        """Prepares data table for piping to Output variable of widget:
        - list of actors in story (1st column),
        - comma-separated list of verbs that each actor is involved in (2nd column)

        Returns:
            data table (pandas dataframe)
        """

        rows = []
        for item in self.noun_action_dict:
            if len(self.noun_action_dict[item]) > 0:
                curr_row = []
                curr_row.append(item)
                curr_row.append(", ".join(list(set(self.noun_action_dict[item]))))
                rows.append(curr_row)

        return pd.DataFrame(rows, columns=["actor", "actions"])

    def __generate_tagging_cache(self, story_elements_df, callback=None):
        result = {}
        c = 1
        for storyid in story_elements_df['storyid'].unique().tolist():
            result[storyid] = {}
            sents_df = story_elements_df[story_elements_df['storyid'] == storyid]
            sorted_df = sents_df.sort_values(by=['sentence_id'], ascending=True)
            sents = sorted_df['sentence'].unique().tolist()
            result[storyid]['000'] = self.__postag_sents(sents, 0, 0, 0, story_elements_df)
            result[storyid]['001'] = self.__postag_sents(sents, 0, 0, 1, story_elements_df)
            result[storyid]['010'] = self.__postag_sents(sents, 0, 1, 0, story_elements_df)
            result[storyid]['011'] = self.__postag_sents(sents, 0, 1, 1, story_elements_df)
            result[storyid]['100'] = self.__postag_sents(sents, 1, 0, 0, story_elements_df)
            result[storyid]['101'] = self.__postag_sents(sents, 1, 0, 1, story_elements_df)
            result[storyid]['110'] = self.__postag_sents(sents, 1, 1, 0, story_elements_df)
            result[storyid]['111'] = self.__postag_sents(sents, 1, 1, 1, story_elements_df)
            c+=1
            if callback:
                increment = ((c/len(story_elements_df['storyid'].unique().tolist()))*80)
                callback(increment)

        return result
    
    def __prepare_story_elements_frame_for_filtering(self, story_elements_df):
        story_elements_df = story_elements_df.copy()
        strcols = ['token_text', 'sentence', 'associated_action']

        for col in strcols:
            story_elements_df[col] = story_elements_df[col].astype(str)

        story_elements_df['storyid'] = 'ST' + story_elements_df['storyid'].astype(str)
        story_elements_df['segment_id'] = 'SE' + story_elements_df['segment_id'].astype(str)

        return story_elements_df
    
    def generate_action_analysis_results(self, story_elements_df, callback=None):
        # self.tagging_cache = self.__generate_tagging_cache(story_elements_df, callback)
        self.num_sents_in_stories = story_elements_df.groupby('storyid')['sentence'].nunique().to_dict()
        story_elements_df = self.__prepare_story_elements_frame_for_filtering(story_elements_df)
        result_df = pd.DataFrame()
        
        word_col = next((word for word in story_elements_df.columns if word.startswith('custom_')), None)
        if word_col is None:
            word_col = 'token_text_lowercase'
            
        past_or_present_tense_verbs_df = story_elements_df[story_elements_df['story_navigator_tag'].isin(['PRES_VB', 'PAST_VB'])]
        result_df = past_or_present_tense_verbs_df.groupby(['storyid', 'segment_id', 'story_navigator_tag'])[word_col].agg(word_col='nunique').reset_index().rename(columns={word_col: "tense_freq"})

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

class ActionMetricCalculator:
    """Unused class / code so far..."""

    def __init__(self, text, listofwords):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        self.stopwords = s
        self.html_result = ""

        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        # self.nlp = self.__load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.0
        self.agent_prominence_score_min = 0.0

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0
