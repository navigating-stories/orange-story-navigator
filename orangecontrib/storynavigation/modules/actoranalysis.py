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
from nltk.tokenize import RegexpTokenizer
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

    PKG = importlib_resources.files(constants.MAIN_PACKAGE)
    NL_STOPWORDS_FILE = (
        PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_STOPWORDS_FILENAME
    )
    NL_PRONOUNS_FILE = (
        PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_PRONOUNS_FILENAME
    )

    def __init__(self, model):
        self.stopwords = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
        self.stopwords = [item for item in self.stopwords if len(item) > 0]
        self.pronouns = self.NL_PRONOUNS_FILE.read_text(encoding="utf-8").split(os.linesep)
        self.pronouns = [item for item in self.pronouns if len(item) > 0]

        self.story_collection = []              # list of story texts that are processed in a session
        self.dataset_level_df_header = []       # column names of dataset (story collection) level dataframe
        self.dataset_level_df = pd.DataFrame()  # complete dataset (story collection) level dataframe
        self.sentence_nlp_models = []           # nlp tagging results for each sentence
        # self.sentences = []                     # sentences in a specific story

        self.html_result = ""                   

        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.active_agency_scores = {}
        self.passive_agency_scores = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        self.custom_category_frequencies = {}

        self.nlp = util.load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.0
        self.agent_prominence_score_min = 0.0

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}
        

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0

    def __update_postagging_metrics(
        self, tagtext, selected_prominence_metric, prominence_score_min, token
    ):
        """After pos-tagging a particular token, this method is executed to calculate the word prominence score
        for the given token and to check whether this score is above the threshold of the user-specified
        minimum word prominence score to display.

        Args:
            tagtext (string): the string representation of the input token from the story text
            selected_prominence_metric: the selected metric by which to calculate the word prominence score

        Returns:
            boolean: True if the word prominence score of the input token is greater or equal to the
            current minimum threshold for the word prominence score specified by the user. False otherwise.
        """

        # This needs to move to Action Analysis module
        vb = util.find_verb_ancestor(token)
        if vb is not None:
            if tagtext in self.noun_action_dict:
                self.noun_action_dict[tagtext].append(vb.text)
            else:
                self.noun_action_dict[tagtext] = []
        # -----------------------------------------------#

        p_score = self.__calculate_prominence_score(tagtext, selected_prominence_metric)
        self.word_prominence_scores[tagtext] = p_score

        if p_score >= prominence_score_min:
            return True
        else:
            return False

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

    def __is_subject(self, tag):
        """Checks whether a given pos-tagged token is a subject of its sentence or not

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            (boolean, string): (True if the given token is a subject of its sentence - False otherwise, the POS tag type of the token)
        """

        if tag[3].lower() in ["nsubj", "nsubj:pass", "csubj"] and tag[1] in [
            "PRON",
            "NOUN",
            "PROPN",
        ]:
            if (
                tag[3].lower() in ["nsubj", "csubj"]
                and util.find_verb_ancestor(tag[4]) is not None
            ):
                if tag[0].lower() in self.active_agency_scores:
                    self.active_agency_scores[tag[0].lower()] += 1
                else:
                    self.active_agency_scores[tag[0].lower()] = 1
                if tag[0].lower() not in self.passive_agency_scores:
                    self.passive_agency_scores[tag[0].lower()] = 0
            else:
                if tag[0].lower() in self.passive_agency_scores:
                    self.passive_agency_scores[tag[0].lower()] += 1
                else:
                    self.passive_agency_scores[tag[0].lower()] = 1
                if tag[0].lower() not in self.active_agency_scores:
                    self.active_agency_scores[tag[0].lower()] = 0

            if tag[1] == "PRON":
                return True, "PRON"
            elif tag[1] == "NOUN":
                return True, "NOUN"
            else:
                return True, "PROPN"
        else:
            if tag[0].lower() in self.passive_agency_scores:
                self.passive_agency_scores[tag[0].lower()] += 1
            else:
                self.passive_agency_scores[tag[0].lower()] = 1
            # if tag[0].lower() not in self.active_agency_scores:
            #     self.active_agency_scores[tag[0].lower()] = 0

        return False, ""

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

    def __find_custom_word_matches(self, custom_word_dict, sentence):
        result = []
        for token in custom_word_dict:
                for word in custom_word_dict[token]:
                    matches = [match.start() for match in re.finditer(r'\b{}\b'.format(re.escape(word)), sentence, flags=re.IGNORECASE)]
                    for match in matches:
                        current_tag = {"start": match, "end": match+len(word), "label": token.upper()}
                        result.append(current_tag)
                        if token in self.custom_category_frequencies:
                            self.custom_category_frequencies[token] += 1
                        else:
                            self.custom_category_frequencies[token] = 1

        return result

    def __get_custom_tags_list(self, story_elements_df):
        result = []
        index_n = 12
        sliced_df = story_elements_df.iloc[:, index_n+1:]
        print("no of cols: ", len(sliced_df))
        print(sliced_df)
        return result

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
        # pos tags that the user wants to highlight
        pos_tags = []
        
        if nouns:
            pos_tags.append("NSP")
            pos_tags.append("NSNP")
        if subjs:
            pos_tags.append("SP")
            pos_tags.append("SNP")
        if custom:
            pos_tags.extend(self.__get_custom_tags_list(story_elements_df))

        if len(pos_tags) == 0:
            for sentence in sentences:
                doc = {"text": sentence, "ents": []}
                options = {"ents": pos_tags, "colors": constants.COLOR_MAP}
                html += displacy.render(doc, style="ent", options=options, manual=True)
            return html

        story_elements_df = story_elements_df.copy()
        story_elements_df['story_navigator_tag'] = story_elements_df['story_navigator_tag'].astype(str)
        story_elements_df['spacy_tag'] = story_elements_df['spacy_tag'].astype(str) 
        matched_df = story_elements_df[story_elements_df['story_navigator_tag'].isin(pos_tags) | story_elements_df['spacy_tag'].isin(pos_tags)]
        matched_df = matched_df.copy()
        
        matched_df['merged_tags'] = np.where(matched_df['story_navigator_tag'] == '-', matched_df['spacy_tag'], matched_df['story_navigator_tag'])
        matched_df['token_start_idx'] = matched_df['token_start_idx'].astype(str)
        matched_df['token_end_idx'] = matched_df['token_end_idx'].astype(str)
        matched_df['displacy_tag_strings'] = matched_df['token_start_idx'] + ' | ' + matched_df['token_end_idx'] + ' | ' + matched_df['merged_tags']
        order_mapping = {value: index for index, value in enumerate(sentences)}

        for sentence in sentences:
            matched_sent_df = matched_df[matched_df['sentence'] == sentence]
            matched_sent_df = matched_sent_df.copy()
            matched_sent_df.loc[:, 'sorting_key'] = matched_sent_df['sentence'].map(lambda value: order_mapping.get(value, len(sentences)))
            matched_sent_df_sorted = matched_sent_df.sort_values(by='sorting_key').drop('sorting_key', axis=1)

            ents = []
            if len(matched_sent_df_sorted) > 0:
                displacy_tags_list = matched_sent_df_sorted['displacy_tag_strings'].tolist()
                for displacy_tag in displacy_tags_list:
                    dtag = displacy_tag.split(' | ')
                    ents.append({"start": int(float(dtag[0])), "end": int(float(dtag[1])), "label": dtag[2]})

                ents = util.remove_duplicate_tagged_entities(ents)                

            doc = {"text": sentence, "ents": ents}
            options = {"ents": pos_tags, "colors": constants.COLOR_MAP}
            html += displacy.render(doc, style="ent", options=options, manual=True)

        if custom:
            return util.remove_span_tags_except_custom(html)
        else:
            return util.remove_span_tags(html)

    def __is_valid_token(self, token):
        """Verifies if token is valid word

        Args:
            token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

        Returns:
            string, boolean : cleaned token text, True if the input token is a valid word, False otherwise
        """

        word = util.get_normalized_token(token)

        # return word, (word not in list(self.stopwords)) and len(word) > 1
        return word, (word not in self.stopwords) and len(word) > 1

    # def __calculate_word_type_count(self, sents, sent_models):
    #     """Calculates the frequency of mentions for each word in the story:
    #         - Number of times word appears as a subject of a sentence
    #         - Number of times the word appears period

    #     Args:
    #         sents (list): list of all sentences (strings) from the input story
    #         sent_models (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sents'
    #     """

    #     for sent_model in sent_models:
    #         for token in sent_model:
    #             normalised_token, is_valid_token = self.__is_valid_token(token)
    #             tag = (token.text, token.pos_, token.tag_, token.dep_, token)
    #             if is_valid_token:
    #                 is_subj, subj_type = self.__is_subject(tag)
    #                 if is_subj:
    #                     if token.text.lower().strip() in ['dit', 'het', 'die']:
    #                         print()
    #                         print('wtf')
    #                         print()
    #                     if token.text.lower().strip() in self.num_occurences_as_subject:
    #                         self.num_occurences_as_subject[
    #                             token.text.lower().strip()
    #                         ] += 1
    #                     else:
    #                         self.num_occurences_as_subject[
    #                             token.text.lower().strip()
    #                         ] = 1
    #                 else:
    #                     if self.__is_pronoun(tag) or self.__is_noun_but_not_pronoun(
    #                         tag
    #                     ):
    #                         if token.text.lower().strip() in self.num_occurences:
    #                             self.num_occurences[token.text.lower().strip()] += 1
    #                         else:
    #                             self.num_occurences[token.text.lower().strip()] = 1

        # for sent in sents:
        #     if len(sent.split()) > 0:
        #         word = sent.split()[0].lower().strip()
        #         if word in self.pronouns:
        #             if word in self.num_occurences_as_subject:
        #                 self.num_occurences_as_subject[word] += 1
        #             else:
        #                 self.num_occurences_as_subject[word] = 1

    def __find_closest_match(self, word, dictionary):
        """Uses fuzzy string matching to find the closest match in a given dictionary (dict) for an input string

        Args:
            word (string): input word
            dictionary (dict): keys are words, values are numbers (mention frequency)

        Returns:
            word, boolean: string of the best match, True if a match is found above the threshold, False otherwise
        """
        highest_score = -10
        word_with_highest_score = word
        for item in dictionary:
            similarity_score = fuzz.ratio(item, word)
            if similarity_score > highest_score:
                highest_score = similarity_score
                word_with_highest_score = item

        if highest_score > 80:
            return word_with_highest_score, True
        else:
            return word, False

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

    def calculate_metrics_freq_table(self):
        """Prepares data table for piping to Output variable of widget: frequency of words in story

        Returns:
            data table (pandas dataframe)
        """

        rows = []
        n = 20
        res = dict(sorted(self.num_occurences.items(), key=itemgetter(1), reverse=True))

        words = list(res.keys())

        for word in words:
            rows.append([word, self.num_occurences[word]])

        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.FREQ_TABLE_HEADER)

    def calculate_metrics_subjfreq_table(self):
        """Prepares data table for piping to Output variable of widget: frequencies as subjects of words in story

        Returns:
            data table (pandas dataframe)
        """
        rows = []
        n = 20
        res = dict(
            sorted(
                self.num_occurences_as_subject.items(), key=itemgetter(1), reverse=True
            )
        )

        words = list(res.keys())

        for word in words:
            rows.append([word, self.num_occurences_as_subject[word]])

        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.SUBFREQ_TABLE_HEADER)


    def calculate_metrics_customfreq_table(self, df):
        """Prepares data table for piping to Output variable of widget: frequencies of custom tokens by user

        Args:
            df (pandas dataframe): the dataframe of all categories of custom words by the user

        Returns:
            data table (pandas dataframe)
        """
        if df is None:
            return pd.DataFrame([], columns=constants.CUSTOMFREQ_TABLE_HEADER)

        rows = []
        n = 20
        res = dict(
            sorted(
                self.custom_category_frequencies.items(), key=itemgetter(1), reverse=True
            )
        )

        words = list(res.keys())

        for word in words:
            rows.append([word, self.custom_category_frequencies[word], str(util.get_column(df, word))])

        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.CUSTOMFREQ_TABLE_HEADER)


    def calculate_metrics_agency_table(self):
        """Prepares data table for piping to Output variable of widget: agency scores of words in story

        Returns:
            data table (pandas dataframe)
        """
        rows = []
        n = 20
        words = set()
        for item in list(self.num_occurences_as_subject.keys()):
            words.add(item)
        for item2 in list(self.num_occurences.keys()):
            words.add(item2)

        words = list(words)

        for word in words:
            if word not in self.active_agency_scores:
                self.active_agency_scores[word] = 0
            if word not in self.passive_agency_scores:
                self.passive_agency_scores[word] = 0

            agency = self.__calculate_agency(word)
            rows.append([word, agency])

        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.AGENCY_TABLE_HEADER)


class ActorMetricCalculator:
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
