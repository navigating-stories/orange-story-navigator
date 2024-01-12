"""Modules required for tagger widget in Story Navigator.
"""

import os
import sys
import pandas as pd
from operator import itemgetter
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from spacy import displacy
import string
import re
from nltk.tokenize import RegexpTokenizer
from thefuzz import fuzz
from statistics import median

class Tagger:
    """Class to perform NLP tagging of relevant actors and actions in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/
    """

    def __init__(self, lang, text_tuples, custom_tags=None):
        self.text_tuples = text_tuples
        self.custom_tags = custom_tags
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

        self.past_verb_count = {}
        self.present_verb_count = {}

        self.nlp = util.load_spacy_pipeline(self.model)
        self.n = 20 # top n scoring tokens for all metrics
        self.complete_data_columns = ['storyid', 'sentence', 'token_text', 'token_start_idx', 'token_end_idx', 'story_navigator_tag', 'spacy_tag', 'spacy_finegrained_tag', 'spacy_dependency', 'is_pronoun_boolean', 'is_sentence_subject_boolean', 'active_voice_subject_boolean', 'associated_action']
        self.complete_data = self.__process_stories(self.nlp, self.text_tuples)
        # Other counts initialisation
        # self.word_count = 0
        # self.word_count_nostops = 0
        # self.sentence_count = 0
        # self.sentence_count_per_word = {}
        # self.active_agency_scores = {}
        # self.passive_agency_scores = {}
        # self.num_occurences = {}
        # self.num_occurences_as_subject = {}
        # self.noun_action_dict = {}

        # self.custom_category_frequencies = {}

        # Scoring related to agent prominence score
        # self.agent_prominence_score_max = 0.0
        # self.agent_prominence_score_min = 0.0

        # Index of word prominence scores for each word in story
        # self.word_prominence_scores = {}
        # self.sentence_nlp_models = []

        # POS counts initialisation
        # self.noun_count = 0
        # self.verb_count = 0
        # self.adjective_count = 0
    
    def __process_stories(self, nlp, text_tuples):
        collection_df = pd.DataFrame()
        for story_tuple in text_tuples:
            story_df = self.__process_story(story_tuple[1], story_tuple[0], nlp)
            collection_df = pd.concat([collection_df, story_df], axis=0, ignore_index=True)
        return collection_df
    
    def __process_story(self, storyid, story_text, nlp):
        story_df = pd.DataFrame()
        sentences = util.preprocess_text(story_text)

        # generate and store nlp tagged models for each sentence
        tagged_sentences = []
        for sentence in sentences:
            if (len(sentence.split()) > 0): # sentence has at least one word in it
                tagged_sentence = nlp(sentence)
                tagged_sentences.append(tagged_sentence)

        story_df = self.__parse_tagged_story(storyid, sentences, tagged_sentences)
        return story_df
        
    def __parse_tagged_story(self, storyid, sentences, tagged_sentences):
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

            ents = []
            if self.custom_tags is not None: # there are custom tags
                custom_matched_tags = self.__find_custom_word_matches(self.custom_tags, sentence)
                for matched_tag in custom_matched_tags:
                    ents.append(matched_tag)

            for tag, span in zip(tags, spans):
                story_df_row = self.__process_tag(storyid, sentence, tag, span)
                story_df_rows.append(story_df_row)

            # special case: first word in a sent can be a pronoun
            if any(word == first_word_in_sent for word in self.pronouns):
                story_df_rows.append([storyid, sentence, first_word_in_sent, 0, len(first_word_in_sent), "SP", '-', '-', '-', True, True, True])

        story_df = pd.DataFrame(story_df_rows, columns=self.complete_data_columns)
        return story_df
        
    def __process_tag(self, storyid, sentence, tag, span):
        row = None
        if self.__is_valid_token(tag):
            if self.__is_subject(tag):
                vb = util.find_verb_ancestor(tag)
                vb_text = '-'
                if vb is not None:
                    vb_text = vb.text
                if self.__is_pronoun(tag):
                    if self.__is_active_voice_subject(tag):
                        row = [storyid, sentence, tag[0], span[0], span[1], "SP", tag[1], tag[2], tag[3], True, True, True, vb_text]
                    else:
                        row = [storyid, sentence, tag[0], span[0], span[1], "SP", tag[1], tag[2], tag[3], True, True, False, vb_text]
                else:
                    if self.__is_active_voice_subject(tag):
                        row = [storyid, sentence, tag[0], span[0], span[1], "SNP", tag[1], tag[2], tag[3], False, True, True, vb_text]
                    else:
                        row = [storyid, sentence, tag[0], span[0], span[1], "SNP", tag[1], tag[2], tag[3], False, True, False, vb_text]
            else:
                if self.__is_pronoun(tag):
                    vb = util.find_verb_ancestor(tag)
                    vb_text = '-'
                    if vb is not None:
                        vb_text = vb.text
                    if self.__is_active_voice_subject(tag):
                        row = [storyid, sentence, tag[0], span[0], span[1], "NSP", tag[1], tag[2], tag[3], True, False, True, vb_text]
                    else:
                        row = [storyid, sentence, tag[0], span[0], span[1], "NSP", tag[1], tag[2], tag[3], True, False, False, vb_text]
                elif self.__is_noun_but_not_pronoun(tag):
                    vb = util.find_verb_ancestor(tag)
                    vb_text = '-'
                    if vb is not None:
                        vb_text = vb.text
                    if self.__is_active_voice_subject(tag):
                        row = [storyid, sentence, tag[0], span[0], span[1], "NSNP", tag[1], tag[2], tag[3], False, False, True, vb_text]
                    else:
                        row = [storyid, sentence, tag[0], span[0], span[1], "NSNP", tag[1], tag[2], tag[3], False, False, False, vb_text]
                else:
                    row = self.__process_action_tag(storyid, sentence, tag, span)
        return row
    
    def __process_action_tag(self, storyid, sentence, tag, span):
            row = None
            if self.__is_valid_token(tag):
                if ((tag[4].text.lower().strip() in self.past_tense_verbs) or (tag[4].text.lower().strip()[:2] == "ge")) and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # past tense
                    row = [storyid, sentence, tag[0], span[0], span[1], "PAST_VB", tag[1], tag[2], tag[3], False, False, False, '-']
                else:
                    if (tag[4].pos_ == "VERB") and (tag[4].text.lower().strip() not in self.false_positive_verbs):  # present tense
                        row = [storyid, sentence, tag[0], span[0], span[1], "PRES_VB", tag[1], tag[2], tag[3], False, False, False, '-']
                    else:
                        row = [storyid, sentence, tag[0], span[0], span[1], "-", tag[1], tag[2], tag[3], False, False, False, '-']
            return row
    
    def __is_valid_token(self, token):
        """Verifies if token is valid word

        Args:
            token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

        Returns:
            string, boolean : cleaned token text, True if the input token is a valid word, False otherwise
        """

        word = util.get_normalized_token(token)
        return word, (word not in self.stopwords) and len(word) > 1

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

        if ((tag[3].lower() in ["nsubj", "nsubj:pass", "csubj"]) and (tag[1] in ["PRON", "NOUN", "PROPN"])):
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

    # def __find_custom_word_matches(self, custom_word_dict, sentence):
    #     result = []
    #     for token in custom_word_dict:
    #             for word in custom_word_dict[token]:
    #                 matches = [match.start() for match in re.finditer(r'\b{}\b'.format(re.escape(word)), sentence, flags=re.IGNORECASE)]
    #                 for match in matches:
    #                     current_tag = {"start": match, "end": match+len(word), "label": token.upper()}
    #                     result.append(current_tag)
    #                     if token in self.custom_category_frequencies:
    #                         self.custom_category_frequencies[token] += 1
    #                     else:
    #                         self.custom_category_frequencies[token] = 1

    #     return result