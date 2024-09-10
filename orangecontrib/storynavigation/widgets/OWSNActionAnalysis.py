# General imports
import os
import re
import sre_constants
from typing import Any, Iterable, List, Set
import numpy as np
import spacy

# Imports from Qt
from AnyQt.QtCore import (
    QAbstractListModel,
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QItemSelectionRange,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QUrl,
)
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QListView,
    QSizePolicy,
    QSplitter,
    QTableView,
)

# Imports from Orange3
from Orange.data import Variable, Table
from Orange.data.domain import Domain, filter_visible
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecanvas.gui.utils import disconnected
from orangewidget.utils.listview import ListViewSearch
from Orange.data.pandas_compat import table_from_frame

# Imports from other Orange3 add-ons
from orangecontrib.text.corpus import Corpus

# Imports from this add-on
from storynavigation.modules.actionanalysis import ActionTagger
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
import storynavigation.modules.error_handling as error_handling
from thefuzz import fuzz

HTML = """
<!doctype html>
<html>
<head>
<script type="text/javascript" src="resources/jquery-3.1.1.min.js">
</script>
<script type="text/javascript" src="resources/jquery.mark.min.js">
</script>
<script type="text/javascript" src="resources/highlighter.js">
</script>
<meta charset='utf-8'>
<style>

table {{ border-collapse: collapse; }}
mark {{ background: #FFCD28; }}

tr > td {{
    padding-bottom: 3px;
    padding-top: 3px;
}}

body {{
    font-family: Helvetica;
    font-size: 10pt;
}}

.line {{ border-bottom: 1px solid #000; }}
.separator {{ height: 5px; }}

.variables {{
    vertical-align: top;
    padding-right: 10px;
}}

.content {{
    /* Adopted from https://css-tricks.com/snippets/css/prevent-long-urls-from-breaking-out-of-container/ */

    /* These are technically the same, but use both */
    overflow-wrap: break-word;
    word-wrap: break-word;

    -ms-word-break: break-all;
    /* This is the dangerous one in WebKit, as it breaks things wherever */
    word-break: break-all;
    /* Instead use this non-standard one: */
    word-break: break-word;

    /* Adds a hyphen where the word breaks, if supported (No Blink) */
    -ms-hyphens: auto;
    -moz-hyphens: auto;
    -webkit-hyphens: auto;
    hyphens: auto;
}}

.token {{
    padding: 3px;
    border: 1px #B0B0B0 solid;
    margin-right: 5px;
    margin-bottom: 5px;
    display: inline-block;
}}

img {{
    max-width: 100%;
}}

</style>
</head>
<body>
<div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #FFC0CB; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Action | past tense
</mark>
<mark class="entity" style="background: #DB7093; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Action | present tense
</mark>
<mark class="entity" style="background: #C7BFC2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Action | future tense
</mark>
{}
</div>
<br/>
{}
</body>
</html>
"""

SEPARATOR = (
    '<tr class="line separator"><td/><td/></tr><tr class="separator"><td/><td/></tr>'
)


def _count_matches(content: List[str], search_string: str, state: TaskState) -> int:
    """
    Count number of appears of any terms in search_string in content texts.

    Parameters
    ----------
    content
        List of texts where we count appearances
    search_string
        Strings that are searched in texts. This parameter has a format
        term1|term2|term3|...

    Returns
    -------
    Number of all matches of search_string in all texts in content list
    """
    matches = 0
    if search_string:
        regex = re.compile(search_string.strip("|"), re.IGNORECASE)
        for i, text in enumerate(content):
            matches += len(regex.findall(text))
            state.set_progress_value((i + 1) / len(content) * 100)
    return matches


class DocumentListModel(QAbstractListModel):
    """
    Custom model for listing documents. Using custom model since Onrage's
    pylistmodel is too slow for large number of documents
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__visible_data = []
        self.__filter_content = []

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole:
            return self.__visible_data[index.row()]
        elif role == Qt.UserRole:
            return self.__filter_content[index.row()]

    def rowCount(self, parent: QModelIndex = None, *args, **kwargs) -> int:
        return len(self.__visible_data)

    def setup_data(self, data: List[str], content: List[str]):
        self.beginResetModel()
        self.__visible_data = data
        self.__filter_content = content
        self.endResetModel()

    def update_filter_content(self, content: List[str]):
        assert len(content) == len(self.__visible_data)
        self.__filter_content = content

    def get_filter_content(self) -> List[str]:
        return self.__filter_content

    def clear(self):
        self.beginResetModel()
        self.__visible_data = []
        self.__filter_content = []
        self.endResetModel()


class DocumentsFilterProxyModel(QSortFilterProxyModel):
    """Filter model for documents list"""

    __regex = None

    def set_filter_string(self, filter_string: str):
        self.__regex = re.compile(filter_string.strip("|"), re.IGNORECASE)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """Filter document that mathc the filter string"""
        if self.__regex is None:
            # filter is not defined yet - show all
            return True
        else:
            index = self.sourceModel().index(source_row, 0, source_parent)
            content = self.sourceModel().data(index, Qt.UserRole)
            res = self.__regex.search(content)
            return bool(res)


class DocumentTableView(QTableView):
    """TableView that disables unselecting all items"""

    def selectionCommand(
        self, index: QModelIndex, event: QEvent = None
    ) -> QItemSelectionModel.SelectionFlags:
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():  # Click on empty viewport; don't clear
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                # Is the last selected index; do not deselect it
                return QItemSelectionModel.NoUpdate
        if (
            event is not None
            and event.type() == QEvent.MouseMove
            and flags & QItemSelectionModel.ToggleCurrent
        ):
            # Disable ctrl drag 'toggle'; can be made to deselect the last
            # index, would need to keep track of the current selection
            # (selectionModel does this but does not expose it)
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags


class VariableListViewSearch(ListViewSearch):
    """ListViewSearch that disables unselecting all items in the list"""

    def selectionCommand(
        self, index: QModelIndex, event: QEvent = None
    ) -> QItemSelectionModel.SelectionFlags:
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():  # Click on empty viewport; don't clear
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                # Is the last selected index; do not deselect it
                return QItemSelectionModel.NoUpdate
        if (
            event is not None
            and event.type() == QEvent.MouseMove
            and flags & QItemSelectionModel.ToggleCurrent
        ):
            # Disable ctrl drag 'toggle'; can be made to deselect the last
            # index, would need to keep track of the current selection
            # (selectionModel does this but does not expose it)
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags

    def set_selection(self, items: Iterable[Variable]):
        """Set selected items in the list view"""
        model = self.model()
        values = self.model()[:]
        items = [it for it in items if it in values]
        selection = QItemSelection()
        if items:
            for val in items:
                index = values.index(val)
                selection.merge(
                    QItemSelection(model.index(index, 0), model.index(index, 0)),
                    QItemSelectionModel.Select,
                )
        self.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)


class VisibleDomainModel(DomainModel):
    """Domain model that filter only visible features"""

    def set_domain(self, domain):
        if domain is not None:
            domain = Domain(
                filter_visible(domain.attributes),
                class_vars=filter_visible(domain.class_vars),
                metas=filter_visible(domain.metas),
            )
        super().set_domain(domain)

class OWSNActionAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = "Actions"
    description = (
        "Provides tools to support basic narrative analysis for actions in stories."
    )
    icon = "icons/action_analysis_icon.png"
    priority = 13

    class Inputs:
        stories = Input("Stories", Corpus, replaces=["Data"])
        story_elements = Input("Story elements", Table)

    class Outputs:
        # selected_story_results = Output("Action stats: selected", Table)
        story_collection_results = Output("Action stats", Table)
        # selected_customfreq_table = Output("Custom tag stats: selected", Table)
        customfreq_table = Output("Custom tag stats", Table)
        # actor_action_table_selected = Output("Action table: selected", Table)
        actor_action_table_full = Output("Action table", Table)

    class Error(OWWidget.Error):
        wrong_input_for_stories = error_handling.wrong_input_for_stories
        wrong_input_for_elements = error_handling.wrong_input_for_elements
        residual_error = error_handling.residual_error
        
    settingsHandler = DomainContextHandler()
    settings_version = 2
    search_features: List[Variable] = ContextSetting([])
    display_features: List[Variable] = ContextSetting([])
    selected_documents: Set[int] = Setting({0}, schema_only=True)
    regexp_filter = ContextSetting("")
    show_tokens = Setting(False)
    autocommit = Setting(True)

    # POS or NER? radiobutton selection of entity type to highlight
    tag_type = Setting(1)

    # Parts of speech (POS) checkbox selected initialization
    past_vbz = Setting(True)
    present_vbz = Setting(True)
    future_vbz = Setting(True)
    all_pos = Setting(True)
    zero_pos = Setting(False)
    custom = Setting(True)

    # Panels for POS and NER tag types or lists
    postags_box = None

    # original text (not tagged)
    original_text = ""
    sli = None

    # list of colour values for the background highlight for each entity type
    highlight_colors = {}

    # list of POS checkboxes for each POS type
    pos_checkboxes = []

    class Warning(OWWidget.Warning):
        no_feats_search = Msg("No features included in search.")
        no_feats_display = Msg("No features selected for display.")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.actiontagger = ActionTagger(constants.NL_SPACY_MODEL)
        self.stories = None  # initialise list of documents (corpus)
        self.story_elements = None  # initialise tagging information
        self.story_elements_dict = {}
        self.action_results_df = None
        self.selected_action_results_df = None
        self.word_col = None
        self.selected_action_table_df = None
        self.full_action_table_df = None
        self.selected_custom_freq = None
        self.full_custom_freq = None
        self.valid_stories = []

        self.__pending_selected_documents = self.selected_documents

        # Search features
        ex_sel = QListView.ExtendedSelection
        self.search_listbox = sl = VariableListViewSearch(selectionMode=ex_sel)
        sl.setModel(VisibleDomainModel(separators=False))
        sl.selectionModel().selectionChanged.connect(self.search_features_changed)

        # Display features
        self.display_listbox = dl = VariableListViewSearch(selectionMode=ex_sel)
        dl.setModel(VisibleDomainModel(separators=False))
        dl.selectionModel().selectionChanged.connect(self.display_features_changed)

        # POS tag list
        self.postags_box = gui.vBox(
            self.controlArea,
            "Story elements to highlight:",
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
        )

        self.pastvc = gui.checkBox(
            self.postags_box,
            self,
            "past_vbz",
            "Actions - past tense",
            callback=self.pos_selection_changed,
        )
        self.presentvc = gui.checkBox(
            self.postags_box,
            self,
            "present_vbz",
            "Actions - present tense",
            callback=self.pos_selection_changed,
        )
        self.futurevc = gui.checkBox(
            self.postags_box,
            self,
            "future_vbz",
            "Actions - future tense",
            callback=self.pos_selection_changed,
        )

        self.custom_tags = gui.checkBox(
            self.postags_box,
            self,
            "custom",
            "Custom tokens",
            callback=self.pos_selection_changed
        )

        self.custom_tags.setChecked(False)
        self.custom_tags.setEnabled(False)

        self.allc = gui.checkBox(self.postags_box, self, "all_pos", "All")
        self.allc.setChecked(False)
        self.allc.stateChanged.connect(self.on_state_changed_pos)
        self.pos_checkboxes = [self.pastvc, self.presentvc, self.futurevc, self.custom_tags]

        self.controlArea.layout().addWidget(self.postags_box)
        # Auto-commit box
        gui.auto_commit(
            self.controlArea,
            self,
            "autocommit",
            "Send data",
            "Auto send is on",
            orientation=Qt.Horizontal,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
        )

        # Search
        self.filter_input = gui.lineEdit(
            self.mainArea,
            self,
            "regexp_filter",
            orientation=Qt.Horizontal,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            label="RegExp Filter:",
            callback=self.refresh_search,
        )

        # Main area
        self.splitter = QSplitter(orientation=Qt.Horizontal, childrenCollapsible=False)
        # Document list
        self.doc_list = DocumentTableView()
        self.doc_list.setSelectionBehavior(QTableView.SelectRows)
        self.doc_list.setSelectionMode(QTableView.ExtendedSelection)
        self.doc_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.doc_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.doc_list.horizontalHeader().setVisible(False)
        self.splitter.addWidget(self.doc_list)

        self.doc_list_model = DocumentListModel()
        proxy_model = DocumentsFilterProxyModel()
        proxy_model.setSourceModel(self.doc_list_model)
        self.doc_list.setModel(proxy_model)
        self.doc_list.selectionModel().selectionChanged.connect(self.selection_changed)
        # Document contents
        self.doc_webview = gui.WebviewWidget(self.splitter, debug=False)
        self.mainArea.layout().addWidget(self.splitter)

    def on_state_changed_pos(self, state):
        for checkBox in self.pos_checkboxes:
            checkBox.setCheckState(state)

    def copy_to_clipboard(self):
        text = self.doc_webview.selectedText()
        QApplication.clipboard().setText(text)

    def pos_selection_changed(self):
        self.show_docs()

    def rehighlight_entities(self):
        self.show_docs()

    @Inputs.stories
    def set_stories(self, stories=None):
        """Stories expects a Corpus. Because Corpus is a subclass of Table, Orange type checking 
        misses wrongly connected inputs.         
        """
        self.valid_stories = []
        if stories is not None:
            if not isinstance(stories, Corpus):
                self.Error.wrong_input_for_stories()
            else:
                self.stories = stories
                self.Error.clear()
        else:
            self.Error.clear()

        if self.story_elements is not None:
            self.Error.clear()
            self.start(
                self.run, 
                self.story_elements
            )

        self.setup_controls()
        self.doc_list.model().set_filter_string(self.regexp_filter)
        self.list_docs()
        self.show_docs()

    @Inputs.story_elements
    def set_story_elements(self, story_elements=None):
        """Story elements expects a table. Because Corpus is a subclass of Table, Orange type checking 
        misses wrongly connected inputs."""

        self.valid_stories = []
        
        if story_elements is not None:
            if isinstance(story_elements, Corpus): 
                self.Error.wrong_input_for_elements()
            else:
                self.Error.clear()
                self.story_elements = util.convert_orangetable_to_dataframe(story_elements)
                self.actiontagger = ActionTagger(self.story_elements['lang'].tolist()[0])
                self.start(
                    self.run, 
                    self.story_elements
                )
                self.postags_box.setEnabled(True)     
        else:
            self.custom_tags.setChecked(False)
            self.custom_tags.setEnabled(False)
            self.postags_box.setEnabled(False)
            self.Error.clear()

        self.setup_controls()
        self.doc_list.model().set_filter_string(self.regexp_filter)
        self.list_docs()
        self.show_docs()

    def on_done(self, res : int) -> None:
        """When matches count is done show the result in the label"""
        self.n_matches = res if res is not None else "n/a"

        # deal with stories that do not have entry in story elements frame
        if self.stories is not None:
            domain = Domain([], metas=self.display_features)
            metas = []
            for item in self.valid_stories:
                metas.append(item.metas.tolist())
            self.stories = Corpus(domain=domain, metas=np.array(metas))
            self.list_docs()

        self.Outputs.story_collection_results.send(
            table_from_frame(
                self.action_results_df
            )
        )

        self.Outputs.actor_action_table_full.send(
            table_from_frame(
                self.full_action_table_df
            )
        )

        if util.frame_contains_custom_tag_columns(self.story_elements):
            self.Outputs.customfreq_table.send(
                table_from_frame(
                    self.full_custom_freq
                )
            )

    def run(self, story_elements, state: TaskState):
        def advance(progress):
            if state.is_interruption_requested():
                raise InterruptedError
            state.set_progress_value(progress)

        self.story_elements = story_elements
        self.action_results_df = self.actiontagger.generate_action_analysis_results(self.story_elements, callback=advance)

        # deal with stories that do not have any text / entry in story elements: remove them from doc list
        story_elements_grouped_by_story = self.story_elements.groupby('storyid')
        for storyid, story_df in story_elements_grouped_by_story:
            if self.stories is not None:
                self.valid_stories.append(self.stories[int(storyid)])
            self.story_elements_dict[storyid] = story_df

        selected_storyids = []
        otherids = []
        for doc_count, c_index in enumerate(sorted(self.selected_documents)):
            selected_storyids.append('ST' + str(c_index))
            otherids.append(str(c_index))

        self.word_col = next((word for word in self.story_elements.columns if word.startswith('custom_')), None)
        if self.word_col is None:
            self.word_col = 'token_text_lowercase'

        only_actions_df = self.story_elements[~self.story_elements['associated_action_lowercase'].str.contains(r'\?')]
        selected_only_actions_df = only_actions_df[only_actions_df['storyid'].isin(otherids)]

        self.selected_action_results_df = self.action_results_df[self.action_results_df['storyid'].isin(selected_storyids)]
        self.selected_action_results_df = self.selected_action_results_df.drop(columns=['storyid']) # assume single story is selected

        full_action_table_df = only_actions_df.groupby(['associated_action_lowercase', 'story_navigator_tag'])[self.word_col].agg(lambda x: ', '.join(set(x))).reset_index()
        self.full_action_table_df = full_action_table_df.rename(columns={'associated_action_lowercase': 'action', 'story_navigator_tag': 'entities_type', self.word_col : 'entities'})
        selected_action_table_df = selected_only_actions_df.groupby(['associated_action_lowercase', 'story_navigator_tag'])[self.word_col].agg(lambda x: ', '.join(set(x))).reset_index()
        self.selected_action_table_df = selected_action_table_df.rename(columns={'associated_action_lowercase': 'action', 'story_navigator_tag': 'entities_type', self.word_col : 'entities'})

        if util.frame_contains_custom_tag_columns(self.story_elements):
            self.custom_tags.setEnabled(True)
            self.selected_custom_freq = self.actiontagger.calculate_customfreq_table(self.story_elements, selected_stories=otherids)
            self.full_custom_freq = self.actiontagger.calculate_customfreq_table(self.story_elements, selected_stories=None)
        else:
            self.custom_tags.setChecked(False)
            self.custom_tags.setEnabled(False)
        
        return self.action_results_df, self.valid_stories, self.selected_action_results_df, self.selected_action_table_df, self.full_action_table_df, self.selected_custom_freq, self.full_custom_freq

    def reset_widget(self):
        # Corpus
        self.stories = None
        self.story_elements = None
        self.custom_tags.setEnabled(False)
        # Widgets
        self.search_listbox.model().set_domain(None)
        self.display_listbox.model().set_domain(None)
        self.filter_input.clear()
        self.update_info()
        # Models/vars
        self.doc_list_model.clear()
        # Warnings
        self.Warning.clear()
        # WebView
        self.doc_webview.setHtml("")

    def setup_controls(self):
        """Setup controls in control area"""
        if self.stories is not None:
            domain = self.stories.domain

            self.search_listbox.model().set_domain(domain)
            self.display_listbox.model().set_domain(domain)
            self.search_features = self.search_listbox.model()[:]
            self.display_features = self.display_listbox.model()[:]

    def select_variables(self):
        """Set selection to display and search features view boxes"""
        smodel = self.search_listbox.model()
        dmodel = self.display_listbox.model()
        # it can happen that domain handler will set some features that are
        # not part of domain - remove them
        self.search_features = [f for f in self.search_features if f in smodel]
        self.display_features = [f for f in self.display_features if f in dmodel]
        # if no features after removing non-existent, select all - default
        if not self.search_features:
            self.search_features = smodel[:]
        if not self.display_features:
            self.display_features = dmodel[:]
        with disconnected(
            self.search_listbox.selectionModel().selectionChanged,
            self.search_features_changed,
        ):
            self.search_listbox.set_selection(self.search_features)
        with disconnected(
            self.display_listbox.selectionModel().selectionChanged,
            self.display_features_changed,
        ):
            self.display_listbox.set_selection(self.display_features)

    def list_docs(self):
        """List documents into the left scrolling area"""
        if self.stories is not None:
            docs = self.regenerate_docs()
            self.doc_list_model.setup_data(self.stories.titles.tolist(), docs)

    def get_el_story_text(self, df):
        return ' '.join(df['sentence'].unique().tolist())               # Concatenate all unique sentences in a dataframe column into a single story text
    
    def fuzzy_match_text(self, text1, text2):
        return fuzz.ratio(text1, text2)                                 # Fuzzy string matching of two story texts
    
    def find_matching_story_in_story_elements(self, c_index, story_text):
        for storyid, story_df in self.story_elements_dict.items():      # Loop through dataframes for each story (subset of rows of the Elements table)
            el_story_text = self.get_el_story_text(story_df)            # Concatenate the sentences in the current dataframe into a single story string
            score = self.fuzzy_match_text(el_story_text, story_text)    # Check if the current story text is the same as the selected story text
            if score >= 90:
                return int(storyid)                                     # If the stories match, return the Elements storyid (the correct story id)
        return c_index                                                  # Otherwise, return the default storyid given by the doclist model
    
    def get_selected_indexes(self) -> Set[int]:
        m = self.doc_list.model().mapToSource
        result = set()
        for i in self.doc_list.selectionModel().selectedRows():         # Each i represents a new selected story
            c_index = m(i).row()                                        # Get the currently selected story i index (int)
            obj = self.regenerate_docs()[c_index]                       # get the story object at c_index location in the doc_list model, obj (str) : has the structure 'filename path/to/filename.ext story-text'
            story_text = ' '.join(obj.split()[2:])                      # Only select the story text itself from obj (third component)
            sentences = util.preprocess_text(story_text)                # Preprocess story i text to match similar output sentences to Elements table (sentences)
            sen_fullstop = [sen+'.' for sen in sentences]               # Add a fullstop after each sentence
            proc_story_text = ' '.join(sen_fullstop)                    # Concatenate sentences together to create a story string
            correct_story_id = self.find_matching_story_in_story_elements(c_index, proc_story_text)     # Find the matching story in Elements table for story i
            result.add(correct_story_id)                                # Add the correct story_id to the selected documents index
        return result
    # def get_selected_indexes(self) -> Set[int]:
    #     m = self.doc_list.model().mapToSource
    #     return {m(i).row() for i in self.doc_list.selectionModel().selectedRows()}

    def set_selection(self) -> None:
        """
        Select documents in selected_documents attribute in the view
        """
        self.selected_documents = self.__pending_selected_documents
        self.__pending_selected_documents = {0}
        view = self.doc_list
        model = view.model()
        source_model = model.sourceModel()

        selection = QItemSelection()
        self.selected_documents = {
            r for r in self.selected_documents if r < len(self.stories)
        }
        for row in self.selected_documents:
            index = model.mapFromSource(source_model.index(row, 0))
            selection.append(QItemSelectionRange(index, index))
        # don't emit selection change to avoid double call of commit function
        # it is already called from set_data
        with disconnected(
            self.doc_list.selectionModel().selectionChanged, self.selection_changed
        ):
            view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def update_selected_action_results(self):
        if self.action_results_df is not None and len(self.action_results_df) > 0:
            selected_storyids = []
            otherids = []
            for doc_count, c_index in enumerate(sorted(self.selected_documents)):
                selected_storyids.append('ST' + str(c_index))
                otherids.append(str(c_index))

            selected_storyids = list(set(selected_storyids)) # only unique items
            otherids = list(set(otherids))

            self.selected_action_results_df = self.action_results_df[self.action_results_df['storyid'].isin(selected_storyids)]
            self.selected_action_results_df = self.selected_action_results_df.drop(columns=['storyid']) # assume single story is selected

            only_actions_df = self.story_elements[~self.story_elements['associated_action_lowercase'].str.contains(r'\?')]
            selected_only_actions_df = only_actions_df[only_actions_df['storyid'].isin(otherids)]

            selected_action_table_df = selected_only_actions_df.groupby(['associated_action_lowercase', 'story_navigator_tag'])[self.word_col].agg(lambda x: ', '.join(set(x))).reset_index()
            self.selected_action_table_df = selected_action_table_df.rename(columns={'associated_action_lowercase': 'action', 'story_navigator_tag': 'entities_type', self.word_col : 'entities'})

            if util.frame_contains_custom_tag_columns(self.story_elements):
                self.custom_tags.setEnabled(True)
                self.selected_custom_freq = self.actiontagger.calculate_customfreq_table(self.story_elements, selected_stories=otherids)
                self.full_custom_freq = self.actiontagger.calculate_customfreq_table(self.story_elements, selected_stories=None)

                self.Outputs.customfreq_table.send(
                    table_from_frame(
                        self.full_custom_freq
                    )
                )
            else:
                self.custom_tags.setChecked(False)
                self.custom_tags.setEnabled(False)

    def selection_changed(self) -> None:
        """Function is called every time the selection changes"""
        self.update_selected_action_results()
        self.selected_documents = self.get_selected_indexes()
        self.show_docs()

    def show_docs(self):
        if self.stories is None or (not hasattr(self, 'actiontagger')):
            return
        
        if self.selected_action_results_df is None and self.action_results_df is not None:
            self.update_selected_action_results()

        self.Warning.no_feats_display.clear()
        if len(self.display_features) == 0:
            self.Warning.no_feats_display()

        parts = []
        for doc_count, c_index in enumerate(sorted(self.selected_documents)):
            text = ""
            for feature in self.display_features:
                value = str(self.stories[c_index, feature.name])
                self.original_text = str(value)

                if feature.name.lower() == "content" or feature.name.lower() == "text":
                    if len(self.story_elements_dict) > 0:
                        value = self.actiontagger.postag_text(
                            value,
                            self.past_vbz,
                            self.present_vbz,
                            self.future_vbz,
                            self.custom,
                            self.story_elements_dict[str(c_index)]
                        )

                if feature in self.search_features and (len(self.regexp_filter) > 0):
                    value = self.__mark_text(self.original_text)

                if feature.name.lower() != "content" and feature.name.lower() != "text":
                    value = value.replace("\n", "<br/>")

                is_image = feature.attributes.get("type", "") == "image"

                if is_image and value != "?":
                    value = os.path.join(feature.attributes.get("origin", ""), value)
                    value = '<img src="{}"></img>'.format(value)

                if feature.name.lower() == "content" or feature.name.lower() == "text":
                    text += (
                        # f'<tr><td class="variables"><strong>{feature.name}:</strong></td>'
                        f'<td class="content">{value}</td></tr>'
                    )

            parts.append(text)

        joined = SEPARATOR.join(parts)
        html = f"<table>{joined}</table>"
        base = QUrl.fromLocalFile(__file__)
        if ((self.story_elements is not None) and len(self.story_elements.columns) <= 13):
            self.doc_webview.setHtml(HTML.format('',html), base)
        else:
            custom_tag_legend = '<mark class="entity" style="background: #98FB98; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">Custom token</mark>'
            self.doc_webview.setHtml(HTML.format(custom_tag_legend, html), base)

    def __mark_text(self, text):
        search_keyword = self.regexp_filter.strip("|")
        if not search_keyword:
            return text

        try:
            reg = re.compile(search_keyword, re.IGNORECASE | re.MULTILINE)
        except sre_constants.error:
            return text

        matches = list(reg.finditer(text))
        if not matches:
            return text

        text = list(text)
        for m in matches[::-1]:
            text[m.start() : m.end()] = list(
                f'<mark data-markjs="true">{"".join(text[m.start(): m.end()])}</mark>'
            )
        return "".join(text)

    @staticmethod
    def __get_selected_rows(view: QListView) -> List[Variable]:
        rows = view.selectionModel().selectedRows()
        values = view.model()[:]
        return [values[row.row()] for row in sorted(rows, key=lambda r: r.row())]

    def search_features_changed(self):
        self.search_features = self.__get_selected_rows(self.search_listbox)
        if self.stories:
            self.doc_list_model.update_filter_content(self.regenerate_docs())
        self.doc_list.model().invalidateFilter()
        self.refresh_search()

    def display_features_changed(self):
        self.display_features = self.__get_selected_rows(self.display_listbox)
        self.show_docs()

    def regenerate_docs(self) -> List[str]:
        self.Warning.no_feats_search.clear()
        if len(self.search_features) == 0:
            self.Warning.no_feats_search()
            if self.stories is not None:
                return self.stories.documents_from_features(self.display_features)
        else:
            if self.stories is not None:
                return self.stories.documents_from_features(self.search_features)
        return None

    def refresh_search(self):
        if self.stories is not None:
            self.doc_list.model().set_filter_string(self.regexp_filter)
            if not self.selected_documents:
                # when currently selected items are filtered selection is empty
                # select first element in the view in that case
                self.doc_list.setCurrentIndex(self.doc_list.model().index(0, 0))
            self.update_info()
            self.start(
                _count_matches,
                self.doc_list_model.get_filter_content(),
                self.regexp_filter,
            )
            self.show_docs()

    def on_exception(self, ex):
        raise ex

    def update_info(self):
        if self.stories is not None:
            has_tokens = self.stories.has_tokens()
            self.n_matching = f"{self.doc_list.model().rowCount()}/{len(self.stories)}"
            self.n_tokens = sum(map(len, self.stories.tokens)) if has_tokens else "n/a"
            self.n_types = len(self.stories.dictionary) if has_tokens else "n/a"
        else:
            self.n_matching = "n/a"
            self.n_matches = "n/a"
            self.n_tokens = "n/a"
            self.n_types = "n/a"

    @gui.deferred
    def commit(self):
        if self.stories is not None:
            selected_docs = sorted(self.get_selected_indexes())
            mask = np.ones(len(self.stories), bool)
            mask[selected_docs] = 0

    def send_report(self):
        self.report_items(
            (
                ("Query", self.regexp_filter),
                ("Matching documents", self.n_matching),
                ("Matches", self.n_matches),
            )
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.update_splitter()

    def update_splitter(self):
        """
        Update splitter that document list on the left never take more
        than 1/3 of the space. It is only set on showEvent. If user
        later changes sizes it stays as it is.
        """
        w1, w2 = self.splitter.sizes()
        ws = w1 + w2
        if w2 < 2 / 3 * ws:
            self.splitter.setSizes([int(ws * 1 / 3), int(ws * 2 / 3)])

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            f_order = context.values.pop("display_features", None)
            display_idx = context.values.pop("display_indices", [])
            search_ids = context.values.pop("search_indices", [])
            if f_order is not None:
                f_order = f_order[0]
                display_features = [f_order[i] for i in display_idx if i < len(f_order)]
                search_features = [f_order[i] for i in search_ids if i < len(f_order)]
                context.values["display_features"] = (display_features, -3)
                context.values["search_features"] = (search_features, -3)

            # old widget used PerfectDomainContextHandler with MATCH_VALUES_ALL
            # now it uses DomainContextHandler. The difference are:
            # - perfect handler stores values in tuple while domain in dicts
            # - domain context handler store class_vars together with attributes
            #   while perfect handler store them separately
            # - since MATCH_VALUES_ALL was used discrete var values were stored
            #   with var name (replacing them with id for discrete var - 1)
            if hasattr(context, "class_vars"):
                context.attributes = {
                    attr: 1 if isinstance(v, list) else v
                    for attr, v in context.attributes + context.class_vars
                }
                context.metas = dict(context.metas)
                delattr(context, "class_vars")

if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
#     from orangecontrib.text.preprocess import BASE_TOKENIZER
#     corpus_ = Corpus.from_file("book-excerpts")
#     corpus_ = corpus_[:3]
#     corpus_ = BASE_TOKENIZER(corpus_)
    WidgetPreview(OWSNActionAnalysis).run(None)
