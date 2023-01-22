# from Orange.data import Table
# from Orange.data.util import get_unique_names
# from Orange.widgets import gui, widget, settings
# from Orange.widgets.widget import OWWidget, Input, Output, Msg
# from orangecontrib.storynavigation import Network
# from orangecontrib.storynavigation.network import community as cd
# from orangewidget.settings import rename_setting
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from orangecontrib.storynavigation.network.readwrite import read_pajek, transform_data_to_orange_table
# from os.path import join, dirname

# ### ---
# from orangecontrib.text import Corpus
# import pandas
# import stanza
# from Orange.data.pandas_compat import table_from_frame
# from AnyQt.QtWidgets import QWidget
# import os
# from typing import Optional, Set, List, Tuple, Dict, Any
# from Orange.widgets.settings import DomainContextHandler, ContextSetting, Setting

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
import sys

class OWSNDSGDepParser(OWWidget):
    name = 'DSG dep-parser'
    description = 'Digital Story Grammer: Dutch dependency parsing with Stanza'
    icon = 'icons/dsg_stanzadep_icon.png'
    priority = 6430

    run_nlp = None

    class Inputs:
        corpus = Input("Corpus", Corpus, default=True)

    class Outputs:
        table = Output("Table", Table)

    def swap_aux_head(self, sentence_df, child, head, heads_head):
        for i in range(0, len(sentence_df)):
            if sentence_df.at[i, "id"] == head:
                sentence_df.at[i, "head"] = child
            elif sentence_df.at[i, "id"] == child:
                sentence_df.at[i, "head"] = heads_head
            elif sentence_df.at[i, "head"] == head:
                sentence_df.at[i, "head"] = child
        return sentence_df

    def correct_attachments_sentence(self, sentence_df):
        children = {}
        xpos = {}
        upos = {}
        text = {}
        heads = {}
        for i, row in sentence_df.iterrows():
            child = row["id"]
            head = row["head"]
            if head not in children:
                children[head] = []
            children[head].append(child)
            xpos[child] = row["xpos"]
            upos[child] = row["upos"]
            text[child] = row["text"]
            heads[child] = head
        for head in children:
            if head != 0 and not re.search("^WW", xpos[head]):
                for child in children[head]:
                    if re.search("^WW", xpos[child]) and upos[child] == "AUX":
                        sentence_df = self.swap_aux_head(sentence_df, child, head, heads[head])
        return sentence_df

    def correct_attachments_table(self, nlp_table_df):
        sentence_df = pandas.DataFrame([])
        nlp_table_df_out = pandas.DataFrame([])
        last_id = -1
        for i, row in nlp_table_df.iterrows():
            if row["id"] < last_id:
                new_sentence_df = self.correct_attachments_sentence(sentence_df)
                if len(nlp_table_df_out) == 0:
                    nlp_table_df_out = new_sentence_df
                else:
                    nlp_table_df_out = pandas.concat([nlp_table_df_out, new_sentence_df])
                sentence_df = pandas.DataFrame([])
            sentence_df = pandas.concat([sentence_df, pandas.DataFrame([row])], ignore_index=True)
            # sentence_df = sentence_df.append(pandas.DataFrame([row]), ignore_index = True)
            last_id = row["id"]
        if len(sentence_df) > 0:
            new_sentence_df = self.correct_attachments_sentence(sentence_df)
            if len(nlp_table_df_out) == 0:
                nlp_table_df_out = new_sentence_df
            else:
                nlp_table_df_out = pandas.concat([nlp_table_df_out, new_sentence_df])
        return nlp_table_df_out

    def nlp_analysis_to_table(self, nlp_analysis):
        nbr_of_words = 0
        for s in nlp_analysis.sentences:
            for w in s.words:
                print()
                print("word: ", w)
                if nbr_of_words == 0:
                    nlp_table_df = pandas.DataFrame({"id": [w.id], 
                                                "text": [w.text], 
                                                "lemma": [w.lemma],
                                                "upos": [w.upos],
                                                "xpos": [w.xpos],
                                                "feats": [w.feats],
                                                "head": [w.head],
                                                "deprel": [w.deprel],
                                                "deps": [w.deps],
                                                "misc": [w.misc],
                                                "start_char": [w.start_char],
                                                "end_char": [w.end_char],
                                                "parent": [w.parent],
                                                "sent": [w.sent],
                                                })
                else:
                    nlp_table_df.loc[len(nlp_table_df.index)] = [ w.id, w.text, w.lemma, w.upos, w.xpos, w.feats, 
                                                                w.head, w.deprel, w.deps, w.misc, w.start_char, w.end_char, 
                                                                w.parent, w.sent, ]
                nbr_of_words += 1
        return nlp_table_df

    def analyze_letter(self, run_nlp, letter_id):
        text = self.read_file(letter_id)
        print()
        print("running nlp...", letter_id)
        nlp_analysis = run_nlp(text)
        print("finished running nlp.")
        print()
        print()
        print("running nlp analysis...", letter_id)
        nlp_table_df = self.nlp_analysis_to_table(nlp_analysis)
        print("finished running nlp analysis.")
        print()
        print("running correct attachments table...", letter_id)
        nlp_table_df = self.correct_attachments_table(nlp_table_df)
        print("finished running correct attachments table.")
        print()
        return text, nlp_table_df

    def read_file(self, in_file_id):
        return self.corpus.documents[in_file_id]

    def __init__(self):
        super().__init__() 
        self.corpus = None
        # import warnings
        # warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

    @Inputs.corpus
    def set_corpus(self, corpus: Optional[Corpus]):
        run_nlp = stanza.Pipeline(lang='nl', processors='tokenize,lemma,pos,depparse')
        all_nlp_data = pandas.DataFrame([])
        
        if hasattr(self, "corpus"):
            if self.corpus is None:
                print("it is none")
            else:
                print("it is not none")
                if (self.corpus is not corpus):
                    print("it is different")
                else:
                    print("it is the same")

        self.corpus = corpus

        if self.corpus is not None:     
            for letter_id in range(0, len(self.corpus.documents)):
                text, nlp_table_df = self.analyze_letter(run_nlp, letter_id)
                all_nlp_data = pandas.concat([all_nlp_data, nlp_table_df])

        self.Outputs.table.send(table_from_frame(all_nlp_data))

        print()
        print()
        print("NLP analysis:")
        print()
        print(all_nlp_data)
        print()

def main():
    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    # network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'lastfm.net'))
    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'Erdos02.net'))
    #transform_data_to_orange_table(network)
    # WidgetPreview(OWSNDSGRuleset).run(set_graph=network)
    WidgetPreview(OWSNDSGDepParser).run()

if __name__ == "__main__":
    main()


# def main(argv=sys.argv):
#     from AnyQt.QtWidgets import QApplication
#     app = QApplication(list(argv))
#     args = app.arguments()
#     if len(args) > 1:
#         filename = args[1]
#     else:
#         filename = "iris"

#     ow = OWSNDSGDepParser()
#     ow.show()
#     ow.raise_()

#     dataset = Table(filename)
#     ow.set_data(dataset)
#     ow.handleNewSignals()
#     app.exec_()
#     ow.set_data(None)
#     ow.handleNewSignals()
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())