import re

from Orange.data import Table
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from AnyQt.QtWidgets import QSizePolicy, QSplitter
from AnyQt.QtCore import Qt
from Orange.widgets import gui
from Orange.data.pandas_compat import table_from_frame

import storynavigation.modules.constants as constants
from storynavigation.modules.settinganalysis import SettingAnalyzer
import storynavigation.modules.util as util


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
    entity_colors = { "DATE": "lightblue",
                      "EVENT": "salmon",
                      "FAC": "silver",
                      "GPE": "lemonchiffon",
                      "LOC": "lightgreen",
                      "TIME": "thistle", }

    class Inputs:
        story_elements = Input("Story elements", Table)


    class Outputs:
        dataset_level_data = Output('Intermediate settings', Table)


    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories_selected = []
        self.story_elements = []
        size_policy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.controlArea.setSizePolicy(size_policy)

        self.make_language_selection_menu()

        self.splitter = QSplitter(orientation=Qt.Horizontal, childrenCollapsible=False)
        self.doc_webview = gui.WebviewWidget(self.splitter, debug=False)
        self.doc_webview.setHtml("")
        self.mainArea.layout().addWidget(self.splitter)


    def make_language_selection_menu(self):
        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Language",
            value="language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True,
            currentIndex=1,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)
        self.select_language_combo.setEnabled(True)


    @Inputs.story_elements
    def set_story_elements(self, story_elements=None):
        """Story elements expects a table. Because Corpus is a subclass of Table, Orange type checking 
        misses wrongly connected inputs."""

        if story_elements is not None:
            self.story_elements = story_elements
            self.text_tuples = self.make_text_tuples(story_elements)
            self.__action_analyze_setting_wrapper()


    def make_text_tuples(self, story_elements):
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        current_story = ""
        current_story_id = ""
        last_sentence = ""
        text_tuples = []
        for index, row in story_elements_df.iterrows():
            story_id = row["storyid"]
            if story_id != current_story_id:
                if current_story_id != "":
                    text_tuples.append((current_story, current_story_id))
                current_story = ""
                current_story_id = story_id
                last_sentence = ""
            if row["sentence"] != last_sentence:
                current_story += row["sentence"] if current_story == "" else " " + row["sentence"]
                last_sentence = row["sentence"]
        if current_story != "":
            text_tuples.append((current_story, current_story_id))
        return text_tuples


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
             language=self.language,
             n_segments=int(self.n_segments),
             text_tuples=self.text_tuples,
             story_elements=self.story_elements,
             callback=move_progress_bar
        )


    def on_done(self, result) -> None:
        self.__visualize_text_data(self.text_tuples, self.analyzer.settings_analysis)
        self.Outputs.dataset_level_data.send(table_from_frame(self.analyzer.settings_analysis))


    def __make_entity_bar_for_html(self):
        return " ".join(['<mark style="background-color:' +
                           self.entity_colors[entity] +
                           f'">{entity}</mark>' for entity in self.entity_colors ])


    def __insert_entity_color_in_story_text(self, story_text, start, end, label):
        story_text = story_text[:end] + "</mark>" + story_text[end:]
        story_text = (story_text[:start] +
                      '<mark style="background-color:' +
                      self.entity_colors[label] +
                      ';">' +
                      story_text[start:])
        return story_text


    def __add_entity_colors_to_story_text(self, story_text, story_id, settings_analysis):
        for index, row in settings_analysis.loc[
                             settings_analysis["storyid"] == "ST" + str(story_id)].iloc[::-1].iterrows():
           start = int(row["character id"])
           end = start + len(row["text"])
           story_text = self.__insert_entity_color_in_story_text(story_text,
                                                                 start,
                                                                 end,
                                                                 row["label"])
        return story_text


    def __add_paragraphs_to_story_text(self, story_text):
        return "<p>" + re.sub("\n\n", "<p>", story_text)


    def __visualize_text_data(self, text_tuples, settings_analysis):
        html_text = "<html><body>"
        html_text += self.__make_entity_bar_for_html()
        for story_text, story_id in text_tuples:
            story_text = self.__add_entity_colors_to_story_text(story_text, story_id, settings_analysis)
            html_text += "<hr>" + self.__add_paragraphs_to_story_text(story_text)
        html_text += "</body></html>"
        self.doc_webview.setHtml(html_text)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    corpus_ = Corpus.from_file("tests/storynavigator-testdata.tab")
    previewer = WidgetPreview(OWSNSettingAnalysis)

