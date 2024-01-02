import logging
import unittest

from storynavigation.widgets.OWSNActorAnalysis import OWSNActorAnalysis
from orangewidget.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
import storynavigation.modules.constants as constants
from storynavigation.modules.actoranalysis import ActorTagger

class test_owsnactoranalysis(WidgetTest):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create a FileHandler
    file_handler = logging.FileHandler('test_actor_analysis.log')

    # Create a Formatter and set it on the FileHandler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the FileHandler to the logger
    logger.addHandler(file_handler)
    
    def setUp(self):
        self.logger.info("setup started...")
        self.widget = self.create_widget(OWSNActorAnalysis)
        self.logger.info("setup completed.")

    def load(self):
        self.logger.info("loading data...")
        self.widget.corpus = Corpus.from_file('/Users/kodymoodley/Documents/work/nlesc-projects/navigating-stories/orange-story-navigator/orangecontrib/storynavigation/tests/storynavigator-testdata.tab')
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
                self.actortagger = ActorTagger(constants.NL_SPACY_MODEL)
                value = self.actortagger.postag_text(
                    value,
                    True,
                    True,
                    False,
                    {},
                    constants.SELECTED_PROMINENCE_METRIC,
                    0.0
                )
        self.logger.info("tagging completed.")

    # def test_tagging(self):
    #                 self.Outputs.metrics_freq_table.send(
    #                     table_from_frame(
    #                         self.actortagger.calculate_metrics_freq_table()
    #                     )
    #                 )
    #                 self.Outputs.metrics_subfreq_table.send(
    #                     table_from_frame(
    #                         self.actortagger.calculate_metrics_subjfreq_table()
    #                     )
    #                 )
    #                 self.Outputs.metrics_customfreq_table.send(
    #                     table_from_frame(
    #                         self.actortagger.calculate_metrics_customfreq_table(self.word_dict)
    #                     )
    #                 )
    #                 self.Outputs.metrics_agency_table.send(
    #                     table_from_frame(
    #                         self.actortagger.calculate_metrics_agency_table()
    #                     )
    #                 )
        


if __name__ == '__main__':
    unittest.main()
