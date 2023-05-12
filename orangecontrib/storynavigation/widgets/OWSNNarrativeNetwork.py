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

import spacy
import nltk
nltk.download('perluniprops')
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

class OWSNNarrativeNetwork(OWWidget, ConcurrentWidgetMixin):
    name = 'Generate Narrative Network'
    description = 'Generates a network of entities and story units for visualisation'
    icon = 'icons/narrative_network_icon.png'
    priority = 6425

    NL_SPACY_MODEL = "nl_core_news_lg" 

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        edge_data = Output('Edge Data', Table)
        node_data = Output('Node Data', Table)
        network = Output('Network', Network)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

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
        # self.nlp_nl.add_pipe("merge_noun_chunks")
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
        for item in list_of_strings:
            identifiers[item] = idx
            vals.append(idx)
            idx += 1

        node_df['node_id'] = vals
        print("node: ", len(vals))
        
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

    def _generate_network(self, texts):
        text_id = 0
        tmp_data = []
        for i in range(0, len(texts)):
            txt = str(texts[i, 'content'])
            print(len(str(txt)))
            sents = sent_tokenize(txt, language='dutch')
            for sent in sents:
                tagged_sentence = self.nlp_nl(sent)

                actions = []
                subjects = []
                objects = []

                subject = 'O'
                action = 'O'
                object = 'O'
            
                for token in tagged_sentence:
                    if ('WW' in token.tag_.split('|')):
                        actions.append(token.text)
                    if ('N' in token.tag_.split('|')) and token.dep_ in ['nsubj']:
                        subjects.append(token.text)
                    if ('N' in token.tag_.split('|')) and token.dep_ not in ['nsubj']:
                        objects.append(token.text)

                if len(subjects) > 0:
                    subject = subjects[0]
                if len(actions) > 0:
                    action = actions[0]
                if len(objects) > 0:
                    object = objects[0]

                tmp_data.append([text_id, "'" + sent + "'", subject, action, object])

            text_id += 1

        # encode categorical data (subject, object, verb strings) into numerical identifiers
        # this is required in order to generate network data that is in the format expected
        # by the orange-network addon
        tmp_data_e, tmp_data_n = self.encode_data(tmp_data) 
        # create a datafame out of the data
        edge_data_tmp = pd.DataFrame(tmp_data, columns = ['story_id', 'sentence', 'subject', 'action', 'object'])
        # edge_data_tmp = pd.DataFrame(tmp_data_e, columns = ['story_id', 'sentence_id', 'subject_id', 'action_id', 'object_id'])
        node_data_tmp = pd.DataFrame(tmp_data_n, columns = ['label', 'types', 'node_id'])

        # all_data = pd.DataFrame(tmp_data)
        # convert the dataframe to orange table format and set outputs of widget
        self.Outputs.edge_data.send(table_from_frame(edge_data_tmp))
        self.Outputs.node_data.send(table_from_frame(node_data_tmp))
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