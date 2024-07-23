from Orange.data import Table
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from AnyQt.QtWidgets import QSizePolicy
from Orange.widgets import gui
from Orange.data.pandas_compat import table_from_frame

import storynavigation.modules.constants as constants
from storynavigation.modules.settinganalysis import SettingAnalyzer


class OWSNSettingAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = 'Setting Analysis'
    description = "Provides tools to help identify the main setting of a story."
    icon = "icons/setting_analysis_icon.png"
    priority = 6480
    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl'
    n_segments = 1


    class Inputs:
        stories_in = Input("Corpus", Corpus, replaces=["Data"])


    class Outputs:
        dataset_level_data = Output('Intermediate settings', Table)


    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories_selected = True
        size_policy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.controlArea.setSizePolicy(size_policy)

        self.make_language_selection_menu()
        self.make_segments_selection_menu()
        self.make_analyze_setting_button()


    def make_language_selection_menu(self):
        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Language",
            value="language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True,
            currentIndex=0,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.select_language_combo.setEnabled(True)
        self.controlArea.layout().addWidget(self.select_language_combo)


    def make_segments_selection_menu(self):
        self.select_n_segments_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Number of segments per story",
            value="n_segments",
            items=constants.N_STORY_SEGMENTS,
            sendSelectedValue=True,
            currentIndex=0,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_n_segments_combo)
        self.select_n_segments_combo.setEnabled(True)


    def make_analyze_setting_button(self):
        self.compute_data_button = gui.button(
            self.controlArea,
            self,
            label="Analyze setting!",
            callback=self.__action_analyze_setting_wrapper,
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


    @Inputs.stories_in
    def set_stories(self, stories_in=None):
        self.stories_selected = [ (text, idx)
            for idx, document in enumerate(stories_in)
                for field in stories_in.domain.metas
                    if (text_field_name := str(field)).lower() in ['text',
                                                                   'content']
                        if len(text := str(stories_in[idx, str(field)])) > 0 ]


    def reset_widget(self):
        self.corpus = None
        self.Warning.clear()


    def __action_analyze_setting_wrapper(self):
        self.start(self.action_analyze_setting)


    def action_analyze_setting(self, state: TaskState):
        def move_progress_bar(progress):
            if state.is_interruption_requested():
                raise InterruptedError
            state.set_progress_value(progress)

        self.analyzer = SettingAnalyzer(
             lang=self.language,
             n_segments=int(self.n_segments),
             text_tuples=self.stories_selected,
             callback=move_progress_bar
        )


    def on_done(self, result) -> None:
        self.Outputs.dataset_level_data.send(table_from_frame(self.analyzer.complete_data))


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    corpus_ = Corpus.from_file("tests/storynavigator-testdata.tab")
    previewer = WidgetPreview(OWSNSettingAnalysis)
    previewer.run(set_stories=corpus_)

