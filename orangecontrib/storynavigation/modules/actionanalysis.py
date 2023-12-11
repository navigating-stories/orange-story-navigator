"""Modules required for Action Analysis widget in Story Navigator.
"""

import sys
import pandas as pd
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

    PKG = importlib_resources.files(constants.MAIN_PACKAGE)
    NL_STOPWORDS_FILE = (
        PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_STOPWORDS_FILENAME
    )
    NL_PRONOUNS_FILE = (
        PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_PRONOUNS_FILENAME
    )

    def __init__(self, model):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        pr = self.NL_PRONOUNS_FILE.read_text(encoding="utf-8")
        self.pronouns = pr
        self.stopwords = s
        self.html_result = ""

        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.active_agency_scores = {}
        self.passive_agency_scores = {}
        self.past_verb_count = {}
        self.present_verb_count = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        self.nlp = util.load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.0
        self.agent_prominence_score_min = 0.0

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}
        self.sentence_nlp_models = []

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0

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

    def postag_text(self, text, past_vbz, present_vbz):
        """POS-tags story text and returns HTML string which encodes the the tagged text, ready for rendering in the UI

        Args:
            text (string): Story text
            nouns (boolean): whether noun tokens should be tagged
            subjs (boolean): whether subject tokens should be tagged
            selected_prominence_metric: the selected metric by which to calculate the word prominence score

        Returns:
            string: HTML string representation of POS tagged text
        """
        sentences = util.preprocess_text(text)
        self.__calculate_pretagging_metrics(sentences)

        # pos tags that the user wants to highlight
        pos_tags = []

        # add pos tags to highlight according to whether the user has selected them or not
        if past_vbz:
            pos_tags.append("PAST_VB")
        if present_vbz:
            pos_tags.append("PRES_VB")

        # output of this function
        html = ""

        # generate and store nlp tagged models for each sentence
        if self.sentence_nlp_models is None or len(self.sentence_nlp_models) == 0:
            # sentence_nlp_models = []
            for sentence in sentences:
                tagged_sentence = self.nlp(sentence)
                self.sentence_nlp_models.append(tagged_sentence)

            self.__calculate_action_type_count(self.sentence_nlp_models)

        # loop through model to filter out those words that need to be tagged (based on user selection and prominence score)
        for sentence, tagged_sentence in zip(sentences, self.sentence_nlp_models):
            tags = []
            tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")
            spans = list(tokenizer.span_tokenize(sentence))

            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_, token.dep_, token))

            ents = []
            for tag, span in zip(tags, spans):
                normalised_token, is_valid_token = self.__is_valid_token(tag)
                if is_valid_token:
                    if tag[4].pos_ == "VERB":
                        vb_tense = tag[4].morph.get("Tense")
                        if vb_tense == "Past":
                            ents.append(
                                {"start": span[0], "end": span[1], "label": "PAST_VB"}
                            )
                        elif vb_tense == "Pres":
                            ents.append(
                                {"start": span[0], "end": span[1], "label": "PRES_VB"}
                            )
                        else:
                            if tag[4].text.lower().strip()[:2] == "ge":  # past tense
                                ents.append(
                                    {
                                        "start": span[0],
                                        "end": span[1],
                                        "label": "PAST_VB",
                                    }
                                )
                            else:
                                ents.append(
                                    {
                                        "start": span[0],
                                        "end": span[1],
                                        "label": "PRES_VB",
                                    }
                                )

                    elif tag[4].pos_ in ["NOUN", "PRON", "PROPN"]:
                        self.__update_postagging_metrics(
                            tag[4].text.lower().strip(), tag[4]
                        )

            # specify sentences and filtered entities to tag / highlight
            doc = {"text": sentence, "ents": ents}

            # specify colors for highlighting each entity type
            colors = {}
            if past_vbz:
                colors["PAST_VB"] = constants.ACTION_PAST_HIGHLIGHT_COLOR
            if present_vbz:
                colors["PRES_VB"] = constants.ACTION_PRESENT_HIGHLIGHT_COLOR

            # collect the above config params together
            options = {"ents": pos_tags, "colors": colors}
            # give all the params to displacy to generate HTML code of the text with highlighted tags
            html += displacy.render(doc, style="ent", options=options, manual=True)

        self.html_result = html
        # return html
        return util.remove_span_tags(html)

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
                    if token.pos_ == "VERB":
                        vb_tense = token.morph.get("Tense")
                        if vb_tense == "Past":
                            if token.text.lower().strip() in self.past_verb_count:
                                self.past_verb_count[token.text.lower().strip()] += 1
                            else:
                                self.past_verb_count[token.text.lower().strip()] = 1
                        elif vb_tense == "Pres":
                            if token.text.lower().strip() in self.present_verb_count:
                                self.present_verb_count[token.text.lower().strip()] += 1
                            else:
                                self.present_verb_count[token.text.lower().strip()] = 1
                        else:
                            if token.text.lower().strip()[:2] == "ge":  # past tense
                                if token.text.lower().strip() in self.past_verb_count:
                                    self.past_verb_count[
                                        token.text.lower().strip()
                                    ] += 1
                                else:
                                    self.past_verb_count[token.text.lower().strip()] = 1
                            else:
                                if (
                                    token.text.lower().strip()
                                    in self.present_verb_count
                                ):
                                    self.present_verb_count[
                                        token.text.lower().strip()
                                    ] += 1
                                else:
                                    self.present_verb_count[
                                        token.text.lower().strip()
                                    ] = 1

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

    def generate_halliday_action_counts_table(self, text, dim_type="realm"):
        rows = []

        # Valid values for 'dim_type' parameter: realm, process, prosub, sub\
        halliday_fname = constants.HALLIDAY_FILENAME.format(dim_type)
        # halliday_fname = "halliday_dimensions_" + dim_type + ".json"
        RESOURCES = ActionTagger.PKG / constants.RESOURCES_SUBPACKAGE
        json_file = RESOURCES.joinpath(halliday_fname).open("r", encoding="utf8")
        halliday_dict = json.load(json_file)

        # Calculate the number of story words in each halliday dimension
        words = text.split()
        halliday_counts = {}
        for item in halliday_dict:
            halliday_counts[item] = 0

        for word in words:
            processed_word = word.lower().strip()
            for item in halliday_dict:
                if processed_word in halliday_dict[item]:
                    halliday_counts[item] += 1

        for item in halliday_dict:
            rows.append([item, halliday_counts[item]])

        return pd.DataFrame(rows, columns=["action", "frequency"])


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
