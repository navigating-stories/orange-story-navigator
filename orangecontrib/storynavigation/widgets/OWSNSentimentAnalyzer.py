from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import Input, OWWidget
from orangecontrib.text.corpus import Corpus
from storynavigation.modules.tagging import Tagger
import storynavigation.modules.constants as constants
from AnyQt.QtWidgets import QSizePolicy
from Orange.data.pandas_compat import table_from_frame
from Orange.data.pandas_compat import table_to_frames
import pandas as pd

class OWSNSentimentAnalyzer(OWWidget, ConcurrentWidgetMixin):
    """Computes positive, negative and neutral sentiment scores for sentences in input stories."""

    name = "Sentiment"
    description = "Uses DistilBERT transformer model"
    icon = "icons/sentiment_analysis_icon.png"
    priority = 14
    keywords = "sentiment analysis"

    class Inputs:
        story_elements = Input("Story elements", Table)

    class Outputs:
        story_elements_with_sentiment = Input("Story elements with sentiment", Table)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl'
    word_column = 'word'
    n_segments = 0 # this selects the the first entry in the list constants.N_STORY_SEGMENTS 

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.story_elements = None # initialise list of documents (corpus)

    @Inputs.stories
    def set_story_elements(self, stories=None):
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
                
        else:
            self.custom_tag_dict = None
            self.select_word_column_combo.clear()
            self.select_word_column_combo.setEnabled(False)

    def reset_widget(self):
        self.stories = None
        self.custom_tag_dict = None
        self.Warning.clear()

    def __generate_dataset_level_data(self):
        n_segments = int(self.n_segments)
        if n_segments == 0: # if the user does not choose explicitly the value in the menu, the value will be 0.
            n_segments = 1 
        if self.stories is not None:
            if len(self.stories) > 0:
                if self.custom_tag_dict is not None:
                    self.tagger = Tagger(
                        lang=self.language, n_segments=n_segments, text_tuples=self.stories, 
                        custom_tags_and_word_column=(self.custom_tag_dict, self.word_column))
                else:
                    self.tagger = Tagger(
                        lang=self.language, n_segments=n_segments, text_tuples=self.stories, 
                        custom_tags_and_word_column=None)

                self.Outputs.dataset_level_data.send(table_from_frame(self.tagger.complete_data))        

if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    # corpus_ = Corpus.from_file("book-excerpts")
    corpus_ = Corpus.from_file("orangecontrib/storynavigation/tests/storynavigator-testdata.tab")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)
    previewer = WidgetPreview(OWSNSentimentAnalyzer)
    # breakpoint()
    previewer.run(set_stories=corpus_, no_exit=True)
    # WidgetPreview(OWSNTagger).run(set_stories=corpus_)