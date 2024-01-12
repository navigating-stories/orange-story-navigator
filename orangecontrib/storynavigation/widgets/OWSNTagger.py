from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from storynavigation.modules.tagging import Tagger
import storynavigation.modules.constants as constants
from AnyQt.QtWidgets import QSizePolicy
from Orange.data.pandas_compat import table_from_frame

class OWSNTagger(OWWidget, ConcurrentWidgetMixin):
    name = 'Tagger'
    description = "Generates part of speech and linguistic tagging information for stories."
    icon = "icons/tagger_icon.png"
    priority = 6424

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])
        custom_tag_dict = Input("Token categories", Table)

    class Outputs:
        dataset_level_data = Output('Story data', Table)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'NL'

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None # initialise list of documents (corpus)
        self.custom_tag_dict = None

        self.settings_panel = gui.vBox(
            self.controlArea,
            "Select language:",
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
        )

        self.select_language_combo = gui.comboBox(
            self.settings_panel,
            self,
            "language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True
        )

        self.compute_data_button = gui.button(
            self.settings_panel, 
            self,
            label="Compute data button",
            callback=self.__generate_dataset_level_data,
            width=160,
            height=60,
            toggleButton=False
        )

        self.select_language_combo.setEnabled(True)
        self.controlArea.layout().addWidget(self.select_language_combo)

    @Inputs.corpus
    def set_corpus(self, corpus=None):
        idx = 0
        self.corpus = []
        for document in corpus:
            text = ''
            for field in corpus.domain.metas:
                text_field_name = str(field)
                if text_field_name.lower() in ['text', 'content']:
                    text = str(corpus[idx, text_field_name])

            if len(text) > 0:
                self.corpus.append((text, idx))

            idx += 1

    @Inputs.custom_tag_dict
    def set_custom_tags(self, custom_tag_dict=None):
        self.custom_tag_dict = custom_tag_dict

    def reset_widget(self):
        self.corpus = None
        self.set_custom_tag_dict = None
        self.Warning.clear()

    def __generate_dataset_level_data(self):
        if self.corpus is not None:
            if len(self.corpus) > 0:
                self.tagger = Tagger(lang=self.language, text_tuples=self.corpus, custom_tags=self.custom_tag_dict)
                self.Outputs.dataset_level_data.send(table_from_frame(self.tagger.complete_data))
                if self.custom_tag_dict is not None:
                    print('Both corpus and custom tags are available!')
                else:
                    print('ONLY corpus is available!')
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