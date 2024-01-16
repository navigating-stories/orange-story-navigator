from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from storynavigation.modules.tagging import Tagger
import storynavigation.modules.constants as constants
from AnyQt.QtWidgets import QSizePolicy
from Orange.data.pandas_compat import table_from_frame
from Orange.data.pandas_compat import table_to_frames
import pandas as pd
import numpy as np

class OWSNTagger(OWWidget, ConcurrentWidgetMixin):
    name = 'Tagger'
    description = "Generates part of speech and linguistic tagging information for stories."
    icon = "icons/tagger_icon.png"
    priority = 6424

    class Inputs:
        stories = Input("Stories", Corpus, replaces=["Data"])
        custom_tag_dict = Input("Custom tags", Table)

    class Outputs:
        dataset_level_data = Output('Story elements', Table)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl'
    word_column = 'word'

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories = None # initialise list of documents (corpus)
        self.custom_tag_dict = None
        self.custom_tag_dict_columns = ['']
        
        size_policy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.controlArea.setSizePolicy(size_policy)

        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Language",
            value="language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)

        self.select_word_column_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Custom tags: select word column",
            value="word_column",
            items=self.custom_tag_dict_columns,
            sendSelectedValue=True,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )

        self.controlArea.layout().addWidget(self.select_word_column_combo)

        self.select_language_combo.setEnabled(True)
        self.select_word_column_combo.setEnabled(True)
        
        self.compute_data_button = gui.button(
            self.controlArea,
            self,
            label="Extract story elements!",
            callback=self.__generate_dataset_level_data,
            width=165,
            height=45,
            toggleButton=False,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            styleSheet="""
                QPushButton {
                    border: 2px solid #555; /* Add border for visibility */
                    border-radius: 10px;
                    padding: 10px;
                }
            """
        )

    @Inputs.stories
    def set_stories(self, stories=None):
        idx = 0
        self.stories = []
        for document in stories:
            text = ''
            for field in stories.domain.metas:
                text_field_name = str(field)
                if text_field_name.lower() in ['text', 'content']:
                    text = str(stories[idx, text_field_name])

            if len(text) > 0:
                self.stories.append((text, idx))

            idx += 1

    @Inputs.custom_tag_dict
    def set_custom_tags(self, custom_tag_dict=None):
        if custom_tag_dict is not None:
            self.custom_tag_dict = pd.concat(table_to_frames(custom_tag_dict), axis=1) # convert the Orange data table to a pandas dataframe, the concat is needed to merge multiple dataframes (Orange splits into multiple frames whenever column uses a different data type)
            if len(self.custom_tag_dict.columns) >= 2:
                self.select_word_column_combo.setEnabled(True)
                self.custom_tag_dict_columns = list(self.custom_tag_dict.columns)
                self.select_word_column_combo.clear()
                self.select_word_column_combo.addItems(self.custom_tag_dict_columns)
                self.word_column = self.custom_tag_dict_columns[0]
        else:
            self.select_word_column_combo.clear()
            self.select_word_column_combo.setEnabled(False)

    def reset_widget(self):
        self.stories = None
        self.custom_tag_dict = None
        self.Warning.clear()

    def __generate_dataset_level_data(self):
        if self.stories is not None:
            if len(self.stories) > 0:
                if self.custom_tag_dict is not None:
                    self.tagger = Tagger(lang=self.language, text_tuples=self.stories, custom_tags_and_word_column=(self.custom_tag_dict, self.word_column))
                    print('Both corpus and custom tags are available!')
                else:
                    self.tagger = Tagger(lang=self.language, text_tuples=self.stories, custom_tags_and_word_column=None)
                    print('ONLY corpus is available!')


                # metas = []
                # for col_name in self.tagger.complete_data.columns:
                #     if str(col_name) in ['story_navigator_tag', 'spacy_tag', 'spacy_finegrained_tag', 'spacy_dependency']:
                #         metas.append({str(col_name) : 'text'})
                #     else:
                #         metas.append({str(col_name) : 'numeric'})

                
                
                # new_vars = []
                # for var in tbl.domain.attributes:
                #     print('var: ', var.name)
                #     print('strvar: ', type(var.name))

                #     if str(var.name) in ['story_navigator_tag', 'spacy_tag', 'spacy_finegrained_tag', 'spacy_dependency']:
                #         new_vars.append(StringVariable(str(var.name)))

                # print()

                # tbl = table_from_frame(self.tagger.complete_data)
                    
                # print()
                # print()
                # print('tbl: ', tbl)
                # print()
                # print()
                # print('test class: ', tbl.domain.class_vars)
                # print()


                # metas=tbl.metas.tolist()
                # print()
                # print()
                # print('tbl1: ', tbl.metas)
                # print()
                # print('tbl2: ', tbl.metas.tolist())
                # print()
                # print()
                # new_vars = []
                # for item in tbl.domain.attributes:
                #     if str(item.name) not in ['story_navigator_tag', 'spacy_tag', 'spacy_finegrained_tag', 'spacy_dependency']:
                #         new_vars.append(item)
                #     else:
                #         for meta in metas:
                #             meta.append(StringVariable(name=str(item.name)))

                # new_domain = Domain(attributes=new_vars, metas=metas)
                # new_tbl = Table.from_table(new_domain, tbl)
                # print()
                # print()
                # print('new_tbl: ', new_tbl)
                # print()
                # print()


                self.Outputs.dataset_level_data.send(table_from_frame(self.tagger.complete_data))
                # self.Outputs.dataset_level_data.send(new_tbl)
                
            else:
                print('Corpus is empty!')
        else:
            if self.custom_tag_dict is not None:
                print('ONLY custom tags are available!')
            else:
                print('BOTH corpus and custom tags are NOT available!')

        

# if __name__ == "__main__":
#     from orangewidget.utils.widgetpreview import WidgetPreview

#     from orangecontrib.text.preprocess import BASE_TOKENIZER

#     corpus_ = Corpus.from_file("book-excerpts")
#     corpus_ = corpus_[:3]
#     corpus_ = BASE_TOKENIZER(corpus_)
#     WidgetPreview(OWSNDSGTagger).run(corpus_)