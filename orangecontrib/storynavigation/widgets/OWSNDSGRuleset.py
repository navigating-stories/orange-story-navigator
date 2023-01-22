from AnyQt.QtCore import QThread, Qt
from AnyQt.QtWidgets import QWidget, QGridLayout
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from orangecontrib.text import Corpus
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import ContinuousVariable, Table, Domain
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.data.pandas_compat import table_from_frame
from typing import Optional
import os
import pandas
import re
import json
import spacy
from spacy.matcher import DependencyMatcher
import sys

class OWSNDSGRuleset(widget.OWWidget):
    name = 'DSG ruleset'
    description = 'Digital Story Grammar: Rules for how to decompose sentences into narrative components'
    icon = 'icons/dsg_ruleset_icon.png'
    priority = 6425

    resizing_enabled = False
    DEBUG = False
    NL_SPACY_PIPELINE = "nl_core_news_sm" 
    NL_DEPENDENCY_PATTERN_FILE = "orangecontrib/storynavigation/widgets/rules/multilingual_dsg_patterns_nl.json"

    class Inputs:
        data = Input("Table", Table)

    class Outputs:
        table = Output("Table", Table)

    auto_commit = Setting(False)

    def __init__(self):
        super().__init__()

        self.data = None
        self.table = None

    def check_dict_in_list(self, dict_obj, dict_list):
        """Check if a dictionary (partially) matches a list of dictionaries.
    
        Note: This function is used to avoid duplicate matches (e.g., Subj+Verb in Subj+Verb+Obj)
    
        Args:
            dict_obj (dict): A dictionary object.
            dict_list (list): A list of dictionary objects.
    
        Returns:
            bool: True if all non-empty items in dict_obj match the items in any dictionary objects in dict_list, otherwise False.
        """
        if dict_obj in dict_list:
            return True
    
        check = [False] * len(dict_obj.keys())
    
        for i, key in enumerate(dict_obj.keys()):
            if str(dict_obj[key]) == "_":
                check[i] = True
                next
            else:
                for ref_dict in dict_list:
                    if dict_obj[key].i == ref_dict[key].i:
                        check[i] = True
                        break
    
        return all(check)

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
    
            if not self.check_dict_in_list(match_dict, matches_list):
                match_dict["match_id"] = match_id
                matches_list.append(match_dict)
    
        return matches_list

    def get_subject_object_verb_table(self, docs, nlp, matcher, keys=["verb", "subj", "obj", "comp", "prep", "aux", "subjadj", "objadj", "obl", "case", "case_arg", "objfixed", ]):
        """Construct a pandas dataframe with subjects, verbs, and objects per sentence of documents.
    
        Args:
            docs (list): A list of text strings.
            nlp (spacy.language.Language): A spacy language pipeline.
            matcher (spacy.matcher.DependencyMatcher): A spacy dependency matcher object.
            keys (list): A list of keys to which the dependency matches are assigned.
                Defaults to subjects, verbs, and objects.
    
        Returns:
            pandas.DataFrame: A dataframe with a row for each match of the dependency matcher and cols:
                doc_id (str): Index of the document in the document list.
                sent_id (str): Index of the sentence in the document.
                sent (spacy.tokens.Span): A spacy span object with the sentence.
                match_id (str): Index of the match in the sentence.
    
                For each key in the `keys` argument:
                    key (spacy.tokens.Token): A spacy token object that matches the dependency matcher patterns.
        """
        docs_piped = nlp.pipe(docs)
    
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
        for i, doc in enumerate(docs_piped): # i: doc index
            if self.DEBUG:
                for token in doc:
                    print(token, token.pos_, token.dep_, token.head)
            for j, sent in enumerate(doc.sents): # j: sent index
                matches = matcher(sent)
                matches_list = self.extract_matches(
                    sent, matches, matcher, nlp, keys=keys)
                for l, match in enumerate(matches_list): # l: match index
                    table_dict["doc_id"].append(str(i))
                    table_dict["sent_id"].append(str(j))
                    table_dict["sent"].append(sent.text)
                    table_dict["match_id"].append(str(match["match_id"]))
    
                    for key in keys:
                        table_dict[key].append(self.append_children_deps(match[key], doc, ["compound", "flat"]))
    
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
                    if self.DEBUG:
                        print("")
    
        for i in range(0, len(table_dict["comp"])):
            # insert table_dict["comp"][i] in table_dict["verb"][i]) here
            pass
    
        return pandas.DataFrame(table_dict)


    def get_children_ids(self, token, children_deps, ids):
        for child in token.children:
            if child.dep_ in children_deps:
                ids.append(child.i)
                ids = self.get_children_ids(child, children_deps, ids)
        return ids


    def append_children_deps(self, token, doc, children_deps):
        """Append children to a token based on dependency tag.
    
        Note: This function is used to append words of a noun compound.
    
        Args:
            token (spacy.token.Token): A spacy token object.
            doc (spacy.token.Doc): A spacy doc object that includes the token.
            children_deps (list): A list of dependency tags.
    
        Returns:
            spacy.token.Token: A span of spacy tokens (token argument plus children with specified dependency tags)
            if token argument is non-empty, the token argument otherwise.
        """
    
        if str(token) != "_":
            children_match_idx = self.get_children_ids(token, children_deps, [token.i])
            span = doc[min(children_match_idx):max(children_match_idx)+1]
    
            return span
        else:
            return ""


    def combine_rows(self, result_table):
        near_duplicate_successive_rows = {}
        for i, row in result_table.iterrows():
            if i > 0:
                different_columns = []
                for column_name in result_table.loc[i].keys():
                    if str(result_table.loc[i][column_name]) != str(result_table.loc[i-1][column_name]):
                        different_columns.append(column_name)
                if len(different_columns) == 1:
                    near_duplicate_successive_rows[i-1] = different_columns[0]
        for i, row in result_table.iterrows():
            if i in near_duplicate_successive_rows:
                row[near_duplicate_successive_rows[i]] = str(row[near_duplicate_successive_rows[i]]) + " " + str(result_table.loc[i+1][near_duplicate_successive_rows[i]])
        for i, row in result_table.iterrows():
            if i-1 in near_duplicate_successive_rows:
                result_table = result_table.drop(i)
        return result_table.reset_index(drop=True)


    def add_verb_group_column(self, result_table):
        subjadj_column = []
        objadj_column = []
        rows_to_be_deleted = []
        for i, row in result_table.iterrows():
            subjadj_column.append(str(row["subjadj"]))
            objadj_column.append(str(row["objadj"]))
            if i > 0 and str(row["doc_id"]) == str(result_table.loc[i-1]["doc_id"]) and str(row["subjadj"]) != "":
                subjadj_column[-2] += " " + str(row["subjadj"])
                rows_to_be_deleted.append(i)
            if i > 0 and str(row["doc_id"]) == str(result_table.loc[i-1]["doc_id"]) and str(row["objadj"]) != "":
                objadj_column[-2] += " " + str(row["objadj"])
                rows_to_be_deleted.append(i)
    
        result_table["subjadj"] = subjadj_column
        result_table["objadj"] = objadj_column
        rows_to_be_deleted = list(set(rows_to_be_deleted))
        rows_to_be_deleted.sort()
        for row_id in rows_to_be_deleted:
            result_table = result_table.drop(row_id)
        verb_group_column = []
        subj_extended_column = []
        obj_extended_column = []
        means_column = []
        for i, row in result_table.iterrows():
            verb_group = str(row["verb"])
            subj_extended = str(row["subj"])
            obj_extended = str(row["obj"])
            means = ""
            if str(row["aux"]) != "":
                verb_group = str(row["aux"]) + " " + verb_group
            if str(row["prep"]) != "":
                verb_group += " " + str(row["prep"])
            if str(row["comp"]) != "":
                verb_group += " " + str(row["comp"])
            if str(row["subjadj"]) != "":
                subj_extended = str(row["subjadj"]) + " " + subj_extended
            if str(row["objadj"]) != "":
                obj_extended = str(row["objadj"]) + " " + obj_extended
            if str(row["objfixed"]) != "":
                obj_extended = obj_extended + " " + str(row["objfixed"])
            if str(row["obl"]) != "" and str(row["case"]) != "":
                means = str(row["case"]) + " " + str(row["case_arg"]) + " " + str(row["obl"])
            verb_group_column.append(verb_group)
            subj_extended_column.append(subj_extended)
            obj_extended_column.append(obj_extended)
            means_column.append(means)
        result_table["verb group"] = verb_group_column
        result_table["subj_extended"] = subj_extended_column
        result_table["obj_extended"] = obj_extended_column
        result_table["means"] = means_column
        result_table = result_table[["doc_id", "sent_id", "sent", "match_id", "subj_extended", "verb group", "obj_extended", "means"]]
        return result_table


    def remove_underscores(self, result_table):
        for i, row in result_table.iterrows():
            for column_name in result_table.loc[i].keys():
                if str(result_table.loc[i][column_name]) == "_":
                    result_table.loc[i][column_name] = ""
        return result_table

    @Inputs.data
    def process_data(self, data: Optional[Table]):
        nlp_nl = self.load_spacy_pipeline(self.NL_SPACY_PIPELINE)
        matcher_nl = self.create_matcher(nlp_nl, self.NL_DEPENDENCY_PATTERN_FILE)
        
        if data is not None:
            print()
            print("hello:")
            print()
            print(data[0])
            print()
            print()
            sentences = [ re.sub("\n", " ", str(data[i]["content"])) for i in range(0, len(data)) ]
            # sentences = [ re.sub("\n", " ", str(data[i][data[i].domain.index("Text")])) for i in range(0, len(data)) ] 
            result_table = self.get_subject_object_verb_table(sentences, nlp_nl, matcher_nl)
            result_table = self.remove_underscores(self.combine_rows(result_table))
            result_table_combined = self.add_verb_group_column(result_table)

            # a predefined domain is necessary to get consistently formatted output
            self.Outputs.table.send(table_from_frame(result_table_combined))

def main():
    WidgetPreview(OWSNDSGRuleset).run()


if __name__ == "__main__":
    main()

# test without GUI and loading Orange
# ------------------------------------
# def main(argv=sys.argv):
#     from AnyQt.QtWidgets import QApplication
#     app = QApplication(list(argv))
#     args = app.arguments()
#     if len(args) > 1:
#         filename = args[1]
#     else:
#         filename = "iris"

#     ow = OWSNDSGRuleset()
#     ow.show()
#     ow.raise_()

#     # dataset = Table(filename)
#     # ow.set_data(dataset)
#     # ow.handleNewSignals()
#     app.exec_()
#     # ow.set_data(None)
#     # ow.handleNewSignals()
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())