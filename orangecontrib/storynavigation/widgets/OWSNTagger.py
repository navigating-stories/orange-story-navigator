from Orange.data import Table, Domain, StringVariable
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
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
import numpy as np
import storynavigation.modules.error_handling as error_handling

class OWSNTagger(OWWidget, ConcurrentWidgetMixin):
    name = 'Elements'
    description = "Extracts story elements. I.e., generates part of speech and linguistic tagging information for input stories."
    icon = "icons/tagger_icon.png"
    priority = 11

    class Inputs:
        stories = Input("Stories", Corpus, replaces=["Data"])
        custom_tag_dict = Input("Custom tags", Table)

    class Outputs:
        dataset_level_data = Output('Story elements', Table)

    class Error(OWWidget.Error):
        wrong_story_input_for_elements = error_handling.wrong_story_input_for_elements
        residual_error = error_handling.residual_error

    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl'
    word_column = 'word'
    n_segments = 1 # n_segments initial value remains 2 in widget for some reason 
    remove_stopwords = constants.NO
    use_infinitives = Setting(False)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories = None # initialise list of documents (corpus)
        self.custom_tag_dict = None
        self.custom_tag_dict_columns = ['']
        self.use_infinitives = False
        
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

        self.select_n_segments_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Number of segments per story",
            value="n_segments",
            items=constants.N_STORY_SEGMENTS,
            sendSelectedValue=True,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_n_segments_combo)

        self.remove_stopwords_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Remove stopwords",
            value="remove_stopwords",
            items=constants.YES_NO_WORDS,
            sendSelectedValue=True,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum),
        )
        self.controlArea.layout().addWidget(self.remove_stopwords_combo)

        self.select_language_combo.setEnabled(True)
        self.select_word_column_combo.setEnabled(True)
        self.select_n_segments_combo.setEnabled(True)
        self.remove_stopwords_combo.setEnabled(True)

        self.compute_data_button = gui.button(
            self.controlArea,
            self,
            label="Extract story elements!",
            callback=self.__generate_dataset_level_data,
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
                
        self.infinitives_checkbox = gui.checkBox(
            widget=self.controlArea,
            master=self,
            value='use_infinitives',
            label='Use infinitives to merge custom words',
            callback=self.on_infinitives_changed,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )

        self.controlArea.layout().addWidget(self.infinitives_checkbox)

    @Inputs.stories
    def set_stories(self, stories=None):
        idx = 0
        self.stories = []
        if stories is not None:
            if not isinstance(stories, Corpus):
                self.Error.wrong_story_input_for_elements()
            else:
                self.Error.clear()
                for document in stories:
                    text = ''
                    for field in stories.domain.metas:
                        text_field_name = str(field)
                        if text_field_name.lower() in ['text', 'content']:
                            text = str(stories[idx, text_field_name])

                    if len(text) > 0:
                        self.stories.append((text, idx))

                    idx += 1
        else:
            self.Error.clear()

    @Inputs.custom_tag_dict
    def set_custom_tags(self, custom_tag_dict=None):
        if custom_tag_dict is not None:
            if isinstance(custom_tag_dict, Corpus):
                self.Error.wrong_story_input_for_elements()
            else:
                self.Error.clear()
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
            self.Error.clear()

    def reset_widget(self):
        self.stories = None
        self.custom_tag_dict = None
        self.Warning.clear()

    def on_done(self, result) -> None:
        if "token_text_lowercase" not in self.tagger.complete_data_df.columns:
            # custom word list added: column names unknown: cannot define domain?
            self.Outputs.dataset_level_data.send(table_from_frame(self.tagger.complete_data_df))
        else:
            # specify domain for columns in tagger output
            tagging_domain = Domain(
                attributes=[
                    ContinuousVariable("storyid"),
                    ContinuousVariable("token_start_idx"),
                    ContinuousVariable("token_end_idx"),
                    ContinuousVariable("spacy_head_idx"),
                    ContinuousVariable("associated_action_idx"),
                    ContinuousVariable("sentence_id"),
                    ContinuousVariable("segment_id"),
                    ContinuousVariable("num_words_in_sentence"),
                    DiscreteVariable.make("story_navigator_tag",
                        values=self.tagger.complete_data_df["story_navigator_tag"].astype(str).unique()),
                    DiscreteVariable.make("spacy_tag",
                        values=self.tagger.complete_data_df["spacy_tag"].astype(str).unique()),
                    DiscreteVariable.make("spacy_finegrained_tag",
                        values=self.tagger.complete_data_df["spacy_finegrained_tag"].astype(str).unique()),
                    DiscreteVariable.make("spacy_dependency",
                        values=self.tagger.complete_data_df["spacy_dependency"].astype(str).unique()),
                    DiscreteVariable.make("spacy_ne",
                        values=self.tagger.complete_data_df["spacy_ne"].astype(str).unique()),
                    DiscreteVariable.make("is_pronoun_boolean",
                        values=["0", "1"]),
                    DiscreteVariable.make("is_sentence_subject_boolean",
                        values=["0", "1"]),
                    DiscreteVariable.make("active_voice_subject_boolean",
                        values=["0", "1"]),
                    DiscreteVariable.make("voice",
                        values=["ACTIVE", "PASSIVE"]),
                    DiscreteVariable.make("future_verb",
                        values=["-", "FUTURE_VB"]),
                    DiscreteVariable.make("lang",
                        values=["en", "nl"]),
                ],
                class_vars=[],
                metas=[
                    StringVariable("sentence"),
                    StringVariable("token_text"),
                    StringVariable("spacy_lemma"),
                    StringVariable("spacy_head_text"),
                    StringVariable("associated_action"),
                    StringVariable("token_text_lowercase"),
                    StringVariable("associated_action_lowercase"),
                ]
            )
    
            #self.Outputs.dataset_level_data.send(table_from_frame(self.tagger.complete_data_df))
            # order of columns in second Table.from_list data should be same as in domain definition
            complete_data_df = self.tagger.complete_data_df[["storyid",
                                                             "token_start_idx",
                                                             "token_end_idx",
                                                             "spacy_head_idx",
                                                             "associated_action_idx",
                                                             "sentence_id",
                                                             "segment_id",
                                                             "num_words_in_sentence",
                                                             "story_navigator_tag",
                                                             "spacy_tag",
                                                             "spacy_finegrained_tag",
                                                             "spacy_dependency",
                                                             "spacy_ne",
                                                             "is_pronoun_boolean",
                                                             "is_sentence_subject_boolean",
                                                             "active_voice_subject_boolean",
                                                             "voice",
                                                             "future_verb",
                                                             "lang",
                                                             "sentence",
                                                             "token_text",
                                                             "spacy_lemma",
                                                             "spacy_head_text",
                                                             "associated_action",
                                                             "token_text_lowercase",
                                                             "associated_action_lowercase"]]
            self.Outputs.dataset_level_data.send(Table.from_list(tagging_domain, complete_data_df.values.tolist()))
 

    def on_infinitives_changed(self):
        #add any additional logic here if needed
        pass
    
    def run(self, lang, n_segments, remove_stopwords, text_tuples, tuple, state: TaskState):
        def advance(progress):
            if state.is_interruption_requested():
                raise InterruptedError
            state.set_progress_value(progress)

        self.tagger = Tagger(
            lang=lang, n_segments=n_segments, remove_stopwords=remove_stopwords, text_tuples=text_tuples, 
            custom_tags_and_word_column=tuple, callback=advance,
            use_infinitives=self.use_infinitives)
        
        return self.tagger.complete_data_df

    def __generate_dataset_level_data(self):
        n_segments = int(self.n_segments)
        if self.stories is not None:
            if len(self.stories) > 0:
                if self.custom_tag_dict is not None:
                    self.start(
                        self.run,
                        self.language,
                        n_segments,
                        self.remove_stopwords,
                        self.stories,
                        (self.custom_tag_dict, self.word_column)
                    )
                else:
                    self.start(
                        self.run,
                        self.language,
                        n_segments,
                        self.remove_stopwords,
                        self.stories,
                        None
                    )

if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    corpus_ = Corpus.from_file("tests/storynavigator-testdata.tab")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)
    previewer = WidgetPreview(OWSNTagger)
    previewer.run(set_stories=corpus_, no_exit=True)
