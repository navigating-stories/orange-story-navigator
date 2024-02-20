"""Modules required for tagger widget in the Orange Story Navigator add-on.
"""

import os
import string
import pandas as pd
import math
import numpy as np
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from nltk.tokenize import RegexpTokenizer
from Orange.data.pandas_compat import table_to_frames
import spacy

class Tagger:
    """Class to perform NLP tagging of relevant actors and actions in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        n_segments (int): Number of segments to split each story into.
    """
    def __init__(self, lang, n_segments, text_tuples, custom_tags_and_word_column=None, callback=None):
        self.text_tuples = text_tuples
        self.lang = lang
        self.n_segments = n_segments
        self.custom_tags = None
        self.word_column = None
        self.complete_data_columns = ['storyid', 'sentence', 'token_text', 'token_start_idx', 'token_end_idx', 'story_navigator_tag', 'spacy_tag', 'spacy_finegrained_tag', 'spacy_dependency', 'is_pronoun_boolean', 'is_sentence_subject_boolean', 'active_voice_subject_boolean', 'associated_action']

        if custom_tags_and_word_column is not None:
            self.word_column = custom_tags_and_word_column[1]
            self.custom_tags = custom_tags_and_word_column[0]
        
        self.stopwords = None
        self.pronouns = None
        self.model = None
        self.past_tense_verbs = None
        self.present_tense_verbs = None
        self.false_positive_verbs = None
        self.__setup_required_nlp_resources(self.lang)

        self.nlp = util.load_spacy_pipeline(self.model)
        self.n = 20 # top n scoring tokens for all metrics

        self.complete_data = self.__process_stories(self.nlp, self.text_tuples, callback)

    def __calculate_story_wordcounts(self, collection_df):
        story_sentence_column = collection_df['sentence'].tolist()
        tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+") # word tokenizer
        num_words_in_sentence_column = []
        for sentence in story_sentence_column:
            spans = list(tokenizer.span_tokenize(sentence))
            num_words_in_sentence_column.append(len(spans))
        return num_words_in_sentence_column

    def __process_stories(self, nlp, text_tuples, callback):
        """This function runs the nlp tagging process on a list of input stories and stores the resulting tagging information in a dataframe.

        Args:
            nlp (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sentences'
            text_tuples (list): each element of the list is a binary tuple. The first component is the text of the story (string) and the second component is a number (int) uniquely identifying that story in the given list

        Returns:
            pandas.DataFrame: a dataframe containing all tagging data for all stories in the given list
        """

        collection_df = pd.DataFrame()
        c = 1
        for story_tuple in text_tuples:
            story_df = self.__process_story(story_tuple[1], story_tuple[0], nlp)
            collection_df = pd.concat([collection_df, story_df], axis=0)
            c+=1
            if callback:
                callback((c / len(text_tuples) * 100))

        if self.custom_tags is not None and self.word_column is not None:
            collection_df['custom_' + self.word_column] = collection_df['token_text'].str.lower()
            collection_df['custom_' + self.word_column] = collection_df['custom_' + self.word_column].str.lstrip('0123456789@#$!“"-')
            collection_df = pd.merge(collection_df, self.custom_tags, left_on='custom_' + self.word_column, right_on=self.word_column, how='left')
            collection_df = collection_df.drop(columns=[self.word_column])
        else:
            collection_df['token_text_lowercase'] = collection_df['token_text'].str.lower()

        collection_df['associated_action'] = collection_df['associated_action'].str.lstrip('0123456789@#$!“"-')
        collection_df['associated_action_lowercase'] = collection_df['associated_action'].str.lower()
        lang_col_values = [self.lang] * len(collection_df)
        collection_df['lang'] = lang_col_values
        story_wordcount_values = self.__calculate_story_wordcounts(collection_df)
        collection_df['num_words_in_sentence'] = story_wordcount_values
        
        return collection_df
    
    def __process_story(self, storyid, story_text, nlp):
        """Given a story text, this function preprocesses the text, then runs and stores the tagging information for each sentence in the story in memory. It then uses this information to generate a dataframe synthesising all the tagging information for downstream analysis.

        Args:
            storyid (int): a number uniquely identifying a specific story
            story_text (string): the text of the story referred to by storyid
            nlp (spacy.language.Language): a spacy language model object to use on the input stories

        Returns:
            pandas.DataFrame: a dataframe containing all tagging data for the given story, and a column with the segment.
            The segment is the segment number of the corresponding sentence in a given story.
        """
        story_df = pd.DataFrame()
        sentences = util.preprocess_text(story_text)

        # generate and store nlp tagged models for each sentence
        tagged_sentences = []
        for sentence in sentences:
            if (len(sentence.split()) > 0): # sentence has at least one word in it
                tagged_sentence = nlp(sentence)
                tagged_sentences.append(tagged_sentence)

        story_df = self.__parse_tagged_story(storyid, sentences, tagged_sentences)

        # Append the segment id by sentence 
        # NOTE: join on storyid x sentence id may be better, but for this we'd need to store the sentence id also in story_df
        sentences_df = []
        sentence_id = 0
        for segment_id, group in enumerate(np.array_split(sentences, self.n_segments)):
            for s in group:
                sentences_df.append([storyid, s, sentence_id, segment_id])
                sentence_id += 1

        sentences_df = pd.DataFrame(sentences_df, columns=["storyid", "sentence", "sentence_id", "segment_id"])

        idx_cols = ["storyid", "sentence"]
        story_df = (story_df.
                    set_index(idx_cols).
                    join(sentences_df.loc[:, idx_cols + ["sentence_id", "segment_id"]].
                         set_index(idx_cols)
                         ).
                    reset_index()
                    )

        return story_df
        
    def __parse_tagged_story(self, storyid, sentences, tagged_sentences):
        """Given a list of sentences in a given story and a list of nlp tagging information for each sentence, this function processes and appends nlp tagging data about these sentences to the master output dataframe

        Args:
            storyid (int): a number uniquely identifying a specific story
            sentences (list): a list of strings (each string is a sentence within the story referred to by storyid
            tagged_sentences (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sentences'

        Returns:
            pandas.DataFrame: a dataframe containing all tagging data for the given story
        """
        story_df = pd.DataFrame()
        story_df_rows = []
        for sentence, tagged_sentence in zip(sentences, tagged_sentences):
            first_word_in_sent = sentence.split()[0].lower().strip()
            tags = []
            # store all spacy nlp tags and dependency info for each token in the sentence in a tuple
            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_, token.dep_, token)) # (text, part of speech (POS) tag, fine-grained POS tag, linguistic dependency, the spacy token object itself)

            tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+") # word tokenizer
            spans = list(tokenizer.span_tokenize(sentence)) # generate token spans in sentence (start and end indices)

            for tag, span in zip(tags, spans):
                story_df_row = self.__process_tag(storyid, sentence, tag, span)
                if story_df_row is not None:
                    story_df_rows.append(story_df_row)

            # special case: first word in a sent can be a pronoun
            if any(word == first_word_in_sent for word in self.pronouns):
                tmp_row = [storyid, sentence, first_word_in_sent, 0, len(first_word_in_sent), "SP", '-', '-', '-', True, True, True, self.__lookup_existing_association(first_word_in_sent, sentence, pd.DataFrame(story_df_rows, columns=self.complete_data_columns))]
                story_df_rows.append(tmp_row)

        story_df = pd.DataFrame(story_df_rows, columns=self.complete_data_columns)
        return story_df
        
    def __process_tag(self, storyid, sentence, tag, span):
        """Given a tagged token in a sentence within a specific story, this function processes and appends data about this token to the master output dataframe

        Args:
            storyid (int): a number uniquely identifying a specific story
            sentence (string): a sentence within this story
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

            span (tuple): 2-component tuple. First component is the matching start index in the sentence of the given tag.text. Second component is the matching end index.

        Returns:
            list: list representing a row of the master story elements dataframe
        """
        row = None
        if self.__is_valid_token(tag):
            if self.__is_subject(tag):
                vb = util.find_verb_ancestor(tag)
                vb_text = '-'
                if vb is not None:
                    vb_text = vb.text
                if self.__is_pronoun(tag):
                    if self.__is_active_voice_subject(tag):
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SP", tag[1], tag[2], tag[3], True, True, True, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SP", tag[1], tag[2], tag[3], True, True, True, vb_text]
                    else:
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SP", tag[1], tag[2], tag[3], True, True, False, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SP", tag[1], tag[2], tag[3], True, True, False, vb_text]
                else:
                    if self.__is_active_voice_subject(tag):
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SNP", tag[1], tag[2], tag[3], False, True, True, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SNP", tag[1], tag[2], tag[3], False, True, True, vb_text]
                    else:
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SNP", tag[1], tag[2], tag[3], False, True, False, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "SNP", tag[1], tag[2], tag[3], False, True, False, vb_text]
            else:
                if self.__is_pronoun(tag):
                    vb = util.find_verb_ancestor(tag)
                    vb_text = '-'
                    if vb is not None:
                        vb_text = vb.text
                    if self.__is_active_voice_subject(tag):
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSP", tag[1], tag[2], tag[3], True, False, True, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSP", tag[1], tag[2], tag[3], True, False, True, vb_text]
                    else:
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSP", tag[1], tag[2], tag[3], True, False, False, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSP", tag[1], tag[2], tag[3], True, False, False, vb_text]
                elif self.__is_noun_but_not_pronoun(tag):
                    vb = util.find_verb_ancestor(tag)
                    vb_text = '-'
                    if vb is not None:
                        vb_text = vb.text
                    if self.__is_active_voice_subject(tag):
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSNP", tag[1], tag[2], tag[3], False, False, True, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSNP", tag[1], tag[2], tag[3], False, False, True, vb_text]
                    else:
                        # row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSNP", tag[1], tag[2], tag[3], False, False, False, vb_text] + self.__lookup_custom_tags(tag)
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "NSNP", tag[1], tag[2], tag[3], False, False, False, vb_text]
                else:
                    row = self.__process_action_tag(storyid, sentence, tag, span)
        return row
    
    def __process_action_tag(self, storyid, sentence, tag, span):
        """Given a tagged token in a sentence within a specific story known to represent an action (verb) rather than other types of tokens such as nouns or adjectives, this function processes and appends data about this action to the master output dataframe

        Args:
            storyid (int): a number uniquely identifying a specific story
            sentence (string): a sentence within this story
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

            span (tuple): 2-component tuple. First component is the matching start index in the sentence of the given tag.text. Second component is the matching end index.

        Returns:
            list: list representing a row of the master story elements dataframe
        """
        row = None
        if self.__is_valid_token(tag):
            if self.lang == constants.NL:
                if ((tag[4].text.lower().strip() in self.past_tense_verbs) or (tag[4].text.lower().strip()[:2] == "ge")) and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # past tense
                    row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "PAST_VB", tag[1], tag[2], tag[3], False, False, False, '-']
                else:
                    if (tag[4].pos_ == "VERB") and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # present tense
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "PRES_VB", tag[1], tag[2], tag[3], False, False, False, '-']
                    else:
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "-", tag[1], tag[2], tag[3], False, False, False, '-']
            else:
                if ((tag[4].text.lower().strip() in self.past_tense_verbs) and (tag[4].text.lower().strip() not in self.false_positive_verbs)):  # past tense
                    row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "PAST_VB", tag[1], tag[2], tag[3], False, False, False, '-']
                else:
                    if (tag[4].pos_ == "VERB") and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # present tense
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "PRES_VB", tag[1], tag[2], tag[3], False, False, False, '-']
                    else:
                        row = [storyid, sentence, tag[0], tag[4].idx, tag[4].idx + len(tag[0]), "-", tag[1], tag[2], tag[3], False, False, False, '-']
        return row
    
    def __is_valid_token(self, token):
        """Verifies if token is valid word

        Args:
            token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

        Returns:
            string, boolean : cleaned token text, True if the input token is a valid word, False otherwise
        """
        word = util.get_normalized_token(token)
        return (word not in self.stopwords) and len(word) > 1 and util.is_only_punctuation(word) != '-'

    def __is_subject(self, tag):
        """Checks whether a given pos-tagged token is a subject of its sentence or not

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            boolean: True if the given token is a subject of its sentence - False otherwise
        """
        if ((tag[3].lower() in ["nsubj", "nsubj:pass", "nsubjpass", "csubj"]) and (tag[1] in ["PRON", "NOUN", "PROPN"])):
            return True
        
        return False
    
    def __is_active_voice_subject(self, tag):
        """Checks whether a given pos-tagged token is involved in an active voice subject role in the sentence

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            boolean: True if the given token is an active voice subject of its sentence - False otherwise
        """
        if (tag[3].lower() in ["nsubj"] and (tag[1] in ["PRON", "NOUN", "PROPN"])):
            return True
        return False

    def __is_pronoun(self, tag):
        """Checks whether a given pos-tagged token is a pronoun or not

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            boolean: True if the given token is a pronoun - False otherwise
        """
        if tag[0].lower().strip() == "ik":
            return True
        if tag[0].lower().strip() not in self.stopwords:
            if tag[1] == "PRON":
                if "|" in tag[2]:
                    tmp_tags = tag[2].split("|")
                    if (tmp_tags[1] == "pers" and tmp_tags[2] == "pron") or (
                        tag[0].lower().strip() == "ik"
                    ):
                        return True
        return False

    def __is_noun_but_not_pronoun(self, tag):
        """Checks whether a given pos-tagged token is a non-pronoun noun (or not)

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            boolean: True if the given token is a non-pronoun noun - False otherwise
        """
        if (not self.__is_pronoun(tag)) and (tag[1] in ["NOUN", "PROPN"]):
            return True
        else:
            return False
        
    def __generate_customtag_column_names(self):
        """Creates a Python list of column names for the boolean columns for each custom tag and classification-scheme

        Returns:
            list: list of strings where each string is the name of a boolean column e.g. "is_realm-scheme_doing" or "is_realm-scheme_being" or "is_realm-scheme_sensing"
                    e.g. ["is_realm-scheme_doing", "is_realm-scheme_being", "is_realm-scheme_sensing", ...]
        """
        column_names = []

        for col in self.custom_tags.columns:    # make sure to only look from Column 2 onwards (Column 1 is a list of words, Column 2 onwards are the tag labels)
            if col != self.word_column:
                for col_value in list(set(self.custom_tags[col].tolist())): # get unique values in column
                    column_names.append('is_' + str(col) + '-scheme_' + str(col_value).lower())
        
        return list(set(column_names))
    
    def __flatten_custom_tag_dictionary(self):
        """Creates a Python dictionary where the keys are each column name generated by `__generate_customtag_column_names()` and the values are a list of strings which belong to the category / tag / label represented by the key

        Returns:
            dict: dict where the keys are strings representing a word category / tag / label, and the values are lists (where each element of the list is a string representing a word belong to the specific category / label / tag)
                    e.g. {
                        "is_realm-scheme_doing" : ['voetballen', 'speel', 'spreek', ...],
                        "is_realm-scheme_being" : ['ben', 'weet', 'vertrouw', ...],
                        "is_realm-scheme_sensing" : ['voel', 'denk', 'overwoog', ...],
                        ...
                    }
        """
        flattened_dict = {}
        if self.customtag_column_names is None:
            self.customtag_column_names = self.__generate_customtag_column_names()

        for col_name in self.customtag_column_names:
            orig_col_name = col_name.split('-scheme_')[0].replace('is_', '')
            orig_tag_name = col_name.split('-scheme_')[1]
            matched_df = self.custom_tags[self.custom_tags[orig_col_name] == orig_tag_name]
            matched_words = list(set(matched_df.iloc[:, 0].tolist()))
            flattened_dict[col_name] = matched_words

        return flattened_dict        
    
    def __lookup_custom_tags(self, tag):
        """Creates a Python dictionary where the keys are each column name generated by `__generate_customtag_column_names()` and the values are a list of strings which belong to the category / tag / label represented by the key

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            list: a list where the nth element is a boolean value (either True or False) indicating whether tag.text belongs to the category represented by the nth custom tag column in self.customtag_column_names
                    e.g. [True, False, True, ...]
        """
        if self.custom_tags is None:
            return []

        results = []

        if self.flattened_custom_tags_dictionary is None:
            self.flattened_custom_tags_dictionary = self.__flatten_custom_tag_dictionary()
            
        for entry in self.flattened_custom_tags_dictionary:
            if isinstance(tag, str):
                if tag.lower().strip() in self.flattened_custom_tags_dictionary[entry]:
                    results.append(True)
                else:
                    results.append(False)
            else:
                if tag[0].lower().strip() in self.flattened_custom_tags_dictionary[entry]:
                    results.append(True)
                else:
                    results.append(False)

        return results
    
    def __lookup_existing_association(self, word, sentence, story_elements_frame):
        matching_rows_sent = story_elements_frame[story_elements_frame['sentence'] == sentence]
        matching_rows_word = matching_rows_sent[matching_rows_sent['token_text'].str.lower() == word.lower()]
        possible_actions = list(set(matching_rows_word['associated_action'].tolist()))
        if len(possible_actions) == 0:
            return '-'
        if len(possible_actions) == 1:
            return util.is_only_punctuation(possible_actions[0])
        
        result = '-'
        for action in possible_actions:
            result = util.is_only_punctuation(action)
            if  result != '-':
                continue

        return result

    def __setup_required_nlp_resources(self, lang):
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input stories (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.pronouns = constants.NL_PRONOUNS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.model = constants.NL_SPACY_MODEL
            self.past_tense_verbs = constants.NL_PAST_TENSE_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.present_tense_verbs = constants.NL_PRESENT_TENSE_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.false_positive_verbs = constants.NL_FALSE_POSITIVE_VERB_FILE.read_text(encoding="utf-8").split(os.linesep)
        else:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.pronouns = constants.EN_PRONOUNS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.model = constants.EN_SPACY_MODEL
            self.past_tense_verbs = constants.EN_PAST_TENSE_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.present_tense_verbs = constants.EN_PRESENT_TENSE_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.false_positive_verbs = constants.EN_FALSE_POSITIVE_VERB_FILE.read_text(encoding="utf-8").split(os.linesep)

        self.stopwords = [item for item in self.stopwords if len(item) > 0]
        self.pronouns = [item for item in self.pronouns if len(item) > 0]
        self.past_tense_verbs = [item for item in self.past_tense_verbs if len(item) > 0]
        self.present_tense_verbs = [item for item in self.present_tense_verbs if len(item) > 0]
        self.false_positive_verbs = [item for item in self.false_positive_verbs if len(item) > 0]
