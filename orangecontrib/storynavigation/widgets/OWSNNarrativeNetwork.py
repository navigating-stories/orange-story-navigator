import os
import re
import sre_constants
from typing import Any, Iterable, List, Set
import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame
from orangecontrib.text.corpus import Corpus
from orangecontrib.network.network import Network
import random

import spacy
import nltk
nltk.download('perluniprops')
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from textblob import TextBlob
from textblob_nl import PatternTagger, PatternAnalyzer

class OWSNNarrativeNetwork(OWWidget, ConcurrentWidgetMixin):
    name = '8) Narrative Network'
    description = 'Generates a network of entities and story units for visualisation'
    icon = 'icons/narrative_network_icon.png'
    priority = 6430

    NL_SPACY_MODEL = "nl_core_news_lg" 

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        edge_data = Output('Edge Data', Table)
        node_data = Output('Node Data', Table)
        sentiment_data = Output('Sentiment Data', Table)
        network = Output('Network', Network)


    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.prominence_scores = {}
        self.corpus = None # initialise list of documents (corpus)
        self.nlp_nl = None # initialise spacy model

    def load_spacy_pipeline(self, name):
        """Check if the spacy language pipeline was downloaded and load it.
        Downloads the language pipeline if not available.
    
        Args:
            name (string): Name of the spacy language.
    
        Returns:
            spacy.language.Language: The spacy language pipeline
        """
        if spacy.util.is_package(name):
            nlp = spacy.load(name)
        else:
            os.system(f"spacy download {name}")
            nlp = spacy.load(name)
        return nlp

    @Inputs.corpus
    def set_data(self, corpus=None):
        self.nlp_nl = self.load_spacy_pipeline(self.NL_SPACY_MODEL)
        self.nlp_nl.add_pipe("merge_noun_chunks")
        self.corpus = corpus
        self._generate_network(self.corpus)

    def reset_widget(self):
        self.corpus = None
        self.Warning.clear()

    def encode_data(self, data):
        """ 
        Encodes categorical data (subject, object, verb strings) into numerical identifiers
        this is required in order to generate network data that is in the format expected
        by the orange-network addon

        Parameters
        ----------

        data : list of lists,
            A table of subject, object, action tuples in list of lists format
            (each list in the master list is a row of the table)

        Returns
        -------
        
        result : list of lists,
            The data from the original table, plus four new columns 
            sentence_id, subject_id, object_id and action_id providing encoded
            identifiers for the subject, object and action strings
        """

        # convert data into dataframe
        df = pd.DataFrame(data, columns = ['story_id', 'sentence', 'subject', 'action', 'object'])

        # initialise dictionary of encoded identifiers
        identifiers = {}
        # generate list of unique strings from the table data
        list_of_strings = []
        list_of_entity_types = []

        subjects = list(set(df['subject'].tolist()))
        subject_types = ['subject'] * len(subjects)
        actions = list(set(df['action'].tolist()))
        action_types = ['action'] * len(actions)
        objects = list(set(df['object'].tolist()))
        object_types = ['object'] * len(objects)

        list_of_strings.extend(subjects)
        list_of_strings.extend(actions)
        list_of_strings.extend(objects)
        list_of_entity_types.extend(subject_types)
        list_of_entity_types.extend(action_types)
        list_of_entity_types.extend(object_types)

        node_df = pd.DataFrame()
        node_df['labels'] = list_of_strings
        node_df['types'] = list_of_entity_types

        # encode strings
        idx = 0
        vals = []
        prominence_scores = []
        for item in list_of_strings:
            identifiers[item] = idx
            vals.append(idx)
            num = random.randint(1, 100)
            prominence_scores.append(num)
            self.prominence_scores[item] = num
            idx += 1

        node_df['node_id'] = vals
        # print("node: ", len(vals))
        node_df['prominence_score'] = prominence_scores

        print()
        print()
        print('dictionary!')
        print('------------')
        print(self.prominence_scores)
        print()
        print()

        result = []
        node_labels = []
        for row in data:
            new_row = []
            new_row.append(row[0]) # append story id
            # new_row.append(identifiers[row[1]]) # append sentence encoding
            new_row.append(identifiers[row[2]]) # append subject encoding
            new_row.append(identifiers[row[3]]) # append action encoding
            new_row.append(identifiers[row[4]]) # append object encoding
            result.append(new_row) # add new row with additional columns of data to return variable

        for item in identifiers.keys():
            node_labels.append([identifiers[item], item])

        return result, node_df.to_numpy().tolist()

    def _traverse_ancestors_recursive(self, token, results_s, results_o):
        # Base case: No more ancestors to traverse
        if not token.ancestors:
            return
        
        # Traverse ancestors recursively until 'nsubj' is found or no more ancestors are left
        for ancestor in token.ancestors:
            # print('ancestor: ', ancestor, ' dep: ', ancestor.dep_, ' pos: ', ancestor.pos_)
            if ancestor.dep_ == 'nsubj' or ancestor.dep_ == 'nsubj:pass' or ancestor.dep_ == 'csubj':
                results_s.append(ancestor.text)
            elif ancestor.dep_ == 'obj' or ancestor.dep_ == 'iobj' or ancestor.dep_ == 'obl' or ancestor.dep_ == 'obl:agent':
                results_o.append(ancestor.text)
            self._traverse_ancestors_recursive(ancestor, results_s, results_o)

    def _traverse_children_recursive(self, token, results_s, results_o):
        # Base case: No more ancestors to traverse
        if not token.children:
            return
            
        # Traverse ancestors recursively until 'nsubj' is found or no more ancestors are left
        for child in token.children:
            # print('child: ', child, ' dep: ', child.dep_, ' pos: ', child.pos_)
            if child.dep_ == 'nsubj' or child.dep_ == 'nsubj:pass' or child.dep_ == 'csubj':
                results_s.append(child.text)
            elif child.dep_ == 'obj' or child.dep_ == 'iobj' or child.dep_ == 'obl' or child.dep_ == 'obl:agent':
                results_o.append(child.text)
            self._traverse_children_recursive(child, results_s, results_o)

    def _get_tuples(self, doc, input_word):
        """
        Traverses dependency tree to find subjects or objects associated with input deontic (for the legal obligation)
        """
        verb = input_word #extract_verb_with_aux(sentence, input_word)
        
        # Find the input word in the sentence
        token = None
        for t in doc:
            if t.text == verb.text.lower():
                token = t
                break
        
        if token is None:
            return [], []
        
        results_s_a = []
        results_o_a = []
        results_s_c = []
        results_o_c = []
        self._traverse_ancestors_recursive(token, results_s_a, results_o_a)
        self._traverse_children_recursive(token, results_s_c, results_o_c)

        sv_tuples = []
        vo_tuples = []
        for item in results_s_a + results_s_c:
            sv_tuples.append((item, verb.text))
        for item in results_o_a + results_o_c:
            vo_tuples.append((verb.text, item))

        return sv_tuples, vo_tuples
    
    def _merge_binary_tuplelsts_into_ternary_tuplelst(self, list1, list2):
        merged_list = []
        for tuple1 in list1:
            foundMatch = False
            for tuple2 in list2:
                if tuple1[1] == tuple2[0]:
                    foundMatch = True
                    merged_list.append((tuple1[0], tuple1[1], tuple2[1]))
            if not foundMatch:
                merged_list.append((tuple1[0], tuple1[1], 'O'))

        return merged_list
    
    def filter_top_n_lists(self, list_of_lists, n):
        top_n_values = sorted([sublist[len(sublist)-1] for sublist in list_of_lists], reverse=True)[:n]
        filtered_lists = [sublist for sublist in list_of_lists if sublist[len(sublist)-1] in top_n_values]
        return filtered_lists
    
    def sort_tuple(self, tup):
        lst = len(tup)
        for i in range(0, lst):
            for j in range(0, lst-i-1):
                if (tup[j][1] > tup[j + 1][1]):
                    temp = tup[j]
                    tup[j] = tup[j + 1]
                    tup[j + 1] = temp
        return tup

    def _generate_network(self, texts):
        text_id = 0
        tmp_data = []
        sentiment_network_tuples = []
        for i in range(0, len(texts)):
            txt = str(texts[i, 'content'])
            # print(len(str(txt)))
            sents = sent_tokenize(txt, language='dutch')
            for sent in sents:
                # blob = TextBlob(sent)
                blob = TextBlob(sent, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
                sentiment_scores = blob.sentiment
                tagged_sentence = self.nlp_nl(sent)
                verbs = []
                nouns = []
                for token in tagged_sentence:
                    if ('WW' in token.tag_.split('|')):
                        verbs.append(token)
                        sv_tuples, vo_tuples = self._get_tuples(tagged_sentence, token)
                        svo_tuples = self._merge_binary_tuplelsts_into_ternary_tuplelst(sv_tuples, vo_tuples)
                    elif ('N' in token.tag_.split('|')) or ('pron' in token.tag_.split('|')) or ('ik' in token.text.lower()):
                        print('here!! ', token.text)
                        print(token.tag_)
                        nouns.append((token, token.idx))
                    else:
                        print('sdasds:', token.text)
                        print(token.tag_)

                for item in svo_tuples:
                    tmp_data.append([text_id, "'" + sent + "'", item[0].lower().strip()+'_subj', item[1].lower().strip(), item[2].lower().strip()+'_obj'])

                print()
                print(nouns)
                print()
                nouns = self.sort_tuple(nouns)
                print(nouns)
                print()
                if len(nouns) > 0:
                    sentiment_subject = nouns[0][0].text
                    sentiment_object = nouns[len(nouns)-1][0].text
                    sentiment_network_tuple = [sentiment_subject, sentiment_object, sentiment_scores[0], sentiment_scores[1], random.randint(1, 100)]
                    sentiment_network_tuples.append(sentiment_network_tuple)

            text_id += 1

        # filter only for top 10 prominence scores for subjects
        # sentiment_network_tuples = self.filter_top_n_lists(sentiment_network_tuples, 7)


        print()
        print()
        print('dictionary!')
        print('------------')
        print(self.prominence_scores)
        print()
        print()
        
        # encode categorical data (subject, object, verb strings) into numerical identifiers
        # this is required in order to generate network data that is in the format expected
        # by the orange-network addon
        tmp_data_e, tmp_data_n = self.encode_data(tmp_data) 
        # create a datafame out of the data
        edge_data_tmp = pd.DataFrame(tmp_data, columns = ['story_id', 'sentence', 'subject', 'action', 'object'])
        # edge_data_tmp = pd.DataFrame(tmp_data_e, columns = ['story_id', 'sentence_id', 'subject_id', 'action_id', 'object_id'])
        node_data_tmp = pd.DataFrame(tmp_data_n, columns = ['label', 'types', 'node_id', 'prominence_score'])
        sentiment_data_tmp = pd.DataFrame(sentiment_network_tuples, columns = ['source_entity', 'target_entity', 'polarity', 'subjectivity', 'subj_prom_score'])

        # all_data = pd.DataFrame(tmp_data)
        # convert the dataframe to orange table format and set outputs of widget
        self.Outputs.edge_data.send(table_from_frame(edge_data_tmp))
        self.Outputs.node_data.send(table_from_frame(node_data_tmp))
        self.Outputs.sentiment_data.send(table_from_frame(sentiment_data_tmp))
        # print()
        # print("handoff: ", node_data_tmp['label'].tolist())
        # print()
        items = node_data_tmp['label'].tolist()
        
        # Table.from_list(Domain([], metas=[StringVariable('label')]), node_data_tmp['label'].tolist())
        # items = Table.from_list(Domain([]), node_data_tmp['node_id'].tolist())
        shape = (len(node_data_tmp['label'].tolist()), len(node_data_tmp['label'].tolist()))

        row = []
        col = []
        data = []
        for item in tmp_data:
            source_index_sa = node_data_tmp['label'].tolist().index(item[2])
            target_index_sa = node_data_tmp['label'].tolist().index(item[3])
            row.append(source_index_sa)
            col.append(target_index_sa)
            data.append(1.0)

            source_index_ao = node_data_tmp['label'].tolist().index(item[3])
            target_index_ao = node_data_tmp['label'].tolist().index(item[4])
            row.append(source_index_ao)
            col.append(target_index_ao)
            data.append(1.0)

        row_np = np.array(row)
        col_np = np.array(col)
        data_np = np.array(data)
        edges = sp.csr_matrix((data_np, (row_np, col_np)), shape=shape)
        
        self.Outputs.network.send(Network(items, edges))
        

# if __name__ == "__main__":
#     from orangewidget.utils.widgetpreview import WidgetPreview

#     from orangecontrib.text.preprocess import BASE_TOKENIZER

#     corpus_ = Corpus.from_file("book-excerpts")
#     corpus_ = corpus_[:3]
#     corpus_ = BASE_TOKENIZER(corpus_)
#     WidgetPreview(OWSNDSGTagger).run(corpus_)