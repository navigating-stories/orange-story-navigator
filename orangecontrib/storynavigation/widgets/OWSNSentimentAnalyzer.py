from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from storynavigation.modules.tagging import Tagger
import storynavigation.modules.constants as constants
from AnyQt.QtWidgets import QSizePolicy
from Orange.data.pandas_compat import table_from_frame
from Orange.data.pandas_compat import table_to_frames
import pandas as pd
from orangecontrib.storynavigation.modules import util
from orangecontrib.storynavigation.modules import sentiment

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
        story_elements_with_sentiment = Output("Story elements with sentiment", Table)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    
    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.story_elements = None
        self.sentiment_analyzer = sentiment.SentimentAnalyzer()
        self.story_elements_with_sentiment_df = None

    @Inputs.story_elements
    def set_story_elements(self, story_elements=None):
        if story_elements is not None:
            self.story_elements = util.convert_orangetable_to_dataframe(story_elements)
            
        self.start(
            self.run
        )

    def run(self, state: TaskState):
        def advance(progress):
            if state.is_interruption_requested():
                raise InterruptedError
            state.set_progress_value(progress)

        self.story_elements_with_sentiment_df = self.sentiment_analyzer.compute_sentiment_scores(self.story_elements, callback=advance)

        return self.story_elements_with_sentiment_df

    def on_done(self, res):
        self.Outputs.story_elements_with_sentiment.send(
            table_from_frame(
                self.story_elements_with_sentiment_df
            )
        )

    def reset_widget(self):
        self.story_elements = None
        self.Warning.clear()

if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    # need to change this file to a Data table object containing story elements (import from CSV perhaps?)
    corpus_ = Corpus.from_file("orangecontrib/storynavigation/tests/storynavigator-testdata.tab")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)
    previewer = WidgetPreview(OWSNSentimentAnalyzer)
    # breakpoint()
    previewer.run(set_story_elements=None, no_exit=True)
    # WidgetPreview(OWSNTagger).run(set_stories=corpus_)