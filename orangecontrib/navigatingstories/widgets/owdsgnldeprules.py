from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from orangecontrib.text import Corpus
from AnyQt.QtWidgets import QWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview
import os
from typing import Optional, Set, List, Tuple, Dict, Any
from Orange.widgets.settings import DomainContextHandler, ContextSetting, Setting
from Orange.data.pandas_compat import table_from_frame
import pandas
import stanza
import re
import json
import spacy
from spacy.matcher import DependencyMatcher


class DSGNLDEPRules(OWWidget):
    name = "DSG NL DEP Rules"
    description = "Digital Story Grammar rules for Dutch operating on Stanza dependency parsing analysis"
    category=None
    icon = "icons/mywidget.svg"
    priority = 200
    keywords = []
    settingsHandler = DomainContextHandler()
    settings_version = 1
    DEBUG = False
    NL_SPACY_PIPELINE = "nl_core_news_sm"
    NL_DEPENDENCY_PATTERN_FILE = "rules/multilingual_dsg_patterns_nl.json"

    class Inputs:
        data = Input("Data", Table)


    class Outputs:
        table = Output("Data", Table)


    def __init__(self):
        super().__init__() 
        import warnings
        warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


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


    def create_matcher(self, nlp, pattern_file):
        """Create a spacy dependency matcher.
    
        Args:
            nlp (spacy.language.Language): A spacy language pipeline.
            pattern_file (str): The path to the dependency pattern .json file for the matcher.
    
        Returns:
            spacy.matcher.DependencyMatcher: A spacy dependency matcher object.
        """
        matcher = DependencyMatcher(nlp.vocab, validate=True)
    
        with open(pattern_file, "r") as file:
            patterns = json.load(file)
    
        for i, pattern in enumerate(patterns):
            matcher.add(i, [pattern])
    
        return matcher


    def extract_matches(self, doc, matches, matcher, nlp, keys):
        """Extract the matched tokens for selected keys.
    
        Args:
            doc (spacy.tokens.Doc): A spacy doc object as returned by a spacy language pipeline.
            matches (list): A list of (match_id, token_ids) tuples as returned by a spacy dependency matcher.
            matcher (spacy.matcher.DependencyMatcher): A spacy dependency matcher object.
            nlp (spacy.language.Language): A spacy language pipeline.
            keys (list): A list of keys to which the dependcy matches are assigned.
    
        Returns:
            list: A list of dictionaries that each contain a match of the dependency matcher. 
                Has the same keys as the `keys` argument. Empty keys contain a spacy token with text='_'.
        """
        matches_list = []
    
        for l, (match_id, token_ids) in enumerate(matches):
            match_dict = {}
    
            for key in keys:
                match_dict[key] = nlp("_")[0]
                
            for k, token_id in enumerate(token_ids):
                key = matcher.get(match_id)[1][0][k]["RIGHT_ID"]
                if key in match_dict.keys():
                    match_dict[key] = doc[token_id]
    
            if not check_dict_in_list(match_dict, matches_list):
                match_dict["match_id"] = match_id
                matches_list.append(match_dict)
    
        return matches_list


    @Inputs.data
    def process_data(self, data: Optional[Table]):

        nlp_nl = self.load_spacy_pipeline(self.NL_SPACY_PIPELINE)
        matcher_nl = self.create_matcher(nlp_nl, self.NL_DEPENDENCY_PATTERN_FILE)
        table_dict = {
            "doc_id": [],
            "sent_id": [],
            "sent": [],
            "match_id": [],
            "subj": [],
            "verb": [],
            "obj": [],
            "comp": [],
            "prep": [],
            "aux": [],
            "subjadj": [],
            "objadj": [],
            "obl": [],
            "case": [],
            "case_arg": [],
            "objfixed": [],
        }
 
        for sent in data:
            # convert sent from RowInstance to Doc
            matches = matcher_nl(sent)
            matches_list = self.extract_matches(
                sent, matches, matcher_nl, nlp_nl, keys=keys)
            for l, match in enumerate(matches_list): # l: match index
                table_dict["doc_id"].append(str(i))
                table_dict["sent_id"].append(str(j))
                table_dict["sent"].append(sent.text)
                table_dict["match_id"].append(str(match["match_id"]))

                for key in keys:
                    table_dict[key].append(append_children_deps(match[key], doc, ["compound", "flat"]))

                    # Check for conjuncts, and add table row for each
                    for conj in match[key].conjuncts:
                        table_dict["doc_id"].append(str(i))
                        table_dict["sent_id"].append(str(j))
                        table_dict["sent"].append(sent.text)
                        table_dict["match_id"].append(str("?"))
                        table_dict[key].append(conj)
                        for key_conj in keys:
                            if key != key_conj:
                                table_dict[key_conj].append(match[key_conj])
                if DEBUG:
                    print("")
        self.Outputs.table.send(data)


if __name__ == "__main__":
    WidgetPreview(DSGNLDEPRules).run()
