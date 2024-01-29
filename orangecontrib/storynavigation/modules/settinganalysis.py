"""Modules required for SettingAnalysis widget in the Orange Story Navigator add-on.
"""

import os
import pandas as pd
import numpy as np
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from collections import Counter



class SettingAnalyzer:
    """Class to analyze the setting in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        lang (str): ISO string of the language of the input text.
        n_segments (int): Number of segments to split each story into.
    """
    def __init__(self, lang, n_segments, text_tuples):
        self.text_tuples = text_tuples
        self.n_segments = n_segments

        self.__setup_required_nlp_resources(lang)
        self.nlp = util.load_spacy_pipeline(self.model)
        # self.n = 20 # top n scoring tokens for all metrics

        self.complete_data = self.__process_stories(self.nlp, self.text_tuples)
    

    def __setup_required_nlp_resources(self, lang): # TODO: make fct reusable? it's also used in OWSNTagger
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input stories (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.model = constants.NL_SPACY_MODEL
        else:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").split(os.linesep)
            self.model = constants.EN_SPACY_MODEL

        self.stopwords = [item for item in self.stopwords if len(item) > 0]


    def __process_stories(self, nlp, text_tuples):
        """Run NLP model, lemmatize tokens and collect them in a dataframe (one row per unique lemma).

        Args:
            nlp (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sentences'
            text_tuples (list): each element of the list is a binary tuple. The first component is the text of the story (string) and the second component is a number (int) uniquely identifying that story in the given list

        Returns:
            pandas.DataFrame: a dataframe containing all tagging data for all stories in the given list
        """
        collection_df = pd.DataFrame()
        for story_tuple in text_tuples:
            story_df = self.__process_story(story_tuple[1], story_tuple[0], nlp)
            collection_df = pd.concat([collection_df, story_df], axis=0, ignore_index=True)

        return collection_df
    

    def __process_story(self, storyid, story_text, nlp):
        """Given a story text, this function preprocesses the text, then runs and stores the lemmatized tokens sentence segment in 
        the story in memory. 

        Args:
            storyid (int): a number uniquely identifying a specific story
            story_text (string): the text of the story referred to by storyid
            nlp (spacy.language.Language): a spacy language model object to use on the input stories

        Returns:
            pandas.DataFrame: a dataframe containing, for each story and segment, the unique lemmas and how often they occur in 
            the respective story segment.
        """
        sentences = util.preprocess_text(story_text)

        # generate and store nlp tagged models for each sentence
        tagged_sentences = []
        for sentence in sentences:
            if (len(sentence.split()) > 0): # sentence has at least one word in it
                tagged_sentence = nlp(sentence)
                tagged_sentences.append(tagged_sentence)

        # extract lemmas if they are not a stopword
        collected_lemmas = []
        for idx, (sentence, tagged_sentence) in enumerate(zip(sentences, tagged_sentences)):
            lemmas = []
            for token in tagged_sentence:
                if util.is_valid_token(token, self.stopwords): 
                    lemmas.append(token.lemma_)
            
            lemmas = Counter(lemmas)
            lemmas = pd.DataFrame.from_dict(lemmas, orient="index").reset_index()
            lemmas = lemmas.rename(columns={"index": "lemma", 0: "count"})
            lemmas["sentence_id"] = idx
            lemmas["sentence"] = sentence
            lemmas["storyid"] = storyid
            collected_lemmas.append(lemmas)

        lemmas_df = pd.concat(collected_lemmas)

        # Append the segment id by sentence 
        # TODO: break out this function as well; it is copy-pasted from tagging.py
        sentences_df = []
        sentence_id = 0
        for segment_id, group in enumerate(np.array_split(sentences, self.n_segments)):
            for s in group:
                sentences_df.append([storyid, s, sentence_id, segment_id])
                sentence_id += 1

        sentences_df = pd.DataFrame(sentences_df, columns=["storyid", "sentence", "sentence_id", "segment_id"])

        idx_cols = ["storyid", "sentence"]
        lemmas_df = (lemmas_df.
                    set_index(idx_cols).
                    join(sentences_df.loc[:, idx_cols + ["segment_id"]].
                         set_index(idx_cols)
                         ).
                    reset_index()
                    )

        # aggregate by segment
        lemmas_df = (lemmas_df.
                     drop(columns=["sentence", "sentence_id"]).
                     groupby(["storyid", "segment_id", "lemma"]).
                     agg("sum").
                     reset_index()
                     )
        
        return lemmas_df
        
