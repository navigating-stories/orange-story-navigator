import logging
import unittest
import os

from storynavigation.widgets.OWSNActorAnalysis import OWSNActorAnalysis
from orangewidget.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
import storynavigation.modules.constants as constants
from storynavigation.modules.actoranalysis import ActorTagger
from Orange.data.pandas_compat import table_from_frame
from Orange.data.pandas_compat import table_to_frames

class test_owsnactoranalysis(WidgetTest):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a FileHandler
    file_handler = logging.FileHandler('test_actor_analysis.log')

    # Create a Formatter and set it on the FileHandler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the FileHandler to the logger
    logger.addHandler(file_handler)

    # Current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    def setUp(self):
        self.logger.info("setup started...")
        self.widget = self.create_widget(OWSNActorAnalysis)
        self.actortagger = ActorTagger(constants.NL_SPACY_MODEL)
        self.tagging_completed = False
        self.logger.info("setup completed.")

    def load(self):
        self.logger.info("loading data...")
        self.widget.corpus = Corpus.from_file(os.path.join(self.current_dir, constants.TEST_DATA_FILE_NAME))
        self.send_signal(self.widget.Inputs.corpus, self.widget.corpus)
        self.logger.info("loading completed.")

    def test_tagging(self):
        self.logger.info("tagging started...")
        self.load()

        idx = 0
        for document in self.widget.corpus:
            value = ''

            if 'Text' == str(self.widget.corpus.domain.metas[0]):
                value = str(self.widget.corpus[idx, 'Text'])
            elif 'Content' == str(self.widget.corpus.domain.metas[0]):
                value = str(self.widget.corpus[idx, 'Content'])
            elif 'text' == str(self.widget.corpus.domain.metas[0]):
                value = str(self.widget.corpus[idx, 'text'])
            elif 'content' == str(self.widget.corpus.domain.metas[0]):
                value = str(self.widget.corpus[idx, 'content'])

            idx += 1

            if len(value) > 0:
                value = self.actortagger.postag_text(
                    value,
                    True,
                    True,
                    False,
                    {},
                    constants.SELECTED_PROMINENCE_METRIC,
                    0.0
                )
                self.tagging_completed = True

        self.logger.info("tagging completed.")

    def test_actor_metrics(self):
        self.logger.info("calculating metrics...")
        if self.tagging_completed:
            ft = self.actortagger.calculate_metrics_freq_table()
            sft = self.actortagger.calculate_metrics_subjfreq_table()
            at = self.actortagger.calculate_metrics_agency_table()
            self.logger.info("")
            self.logger.info(ft.head(1))
            self.logger.info("")
            self.logger.info(sft.head(1))
            self.logger.info("")
            self.logger.info(at.head(1))
            self.logger.info("")
            self.logger.info("metrics calculated.")
        else:
            self.logger.info("metrics could not be calculated.")         

if __name__ == '__main__':
    unittest.main()
