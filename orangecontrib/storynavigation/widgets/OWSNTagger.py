from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from storynavigation.modules.tagging import Tagger

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

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None # initialise list of documents (corpus)
        self.custom_tag_dict = None
        self.compute_data_button = gui.button(
            self.controlArea, 
            self,
            label="Compute data button",
            callback=self.__generate_dataset_level_data,
            width=160,
            height=60,
            toggleButton=False
        )

    @Inputs.corpus
    def set_corpus(self, corpus=None):
        self.corpus = corpus

    @Inputs.custom_tag_dict
    def set_custom_tags(self, custom_tag_dict=None):
        self.custom_tag_dict = custom_tag_dict

    def reset_widget(self):
        self.corpus = None
        self.set_custom_tag_dict = None
        self.Warning.clear()

    def __generate_dataset_level_data(self):
        if self.corpus is not None:

            if self.custom_tag_dict is not None:
                print('Both corpus and custom tags are available!')
            else:
                print('ONLY corpus is available!')
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