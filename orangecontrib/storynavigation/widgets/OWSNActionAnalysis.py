# General imports
import os
import re
import sre_constants
from typing import Any, Iterable, List, Set
import numpy as np
import pandas as pd
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
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
from Orange.data.domain import Domain, filter_visible
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecanvas.gui.utils import disconnected
from orangewidget.utils.listview import ListViewSearch
from Orange.data.pandas_compat import table_from_frame, table_to_frames


# Imports from other Orange3 add-ons
from orangecontrib.text.corpus import Corpus

# Imports from this add-on
from storynavigation.modules.actionanalysis import ActionTagger
from storynavigation.modules.tagging import Tagger
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

    def _auto_generate_story_elements(self):
        """
        Generate a story elements dataframe by running the Tagger internally
        when no 'Story elements' input is connected.

        Uses the same defaults as the Elements widget.
        """
        if self.stories is None or len(self.stories) == 0:
            return None

        lang = "nl"
        try:
            df, _ = table_to_frames(self.stories)
            if "lang" in df.columns:
                lang_series = df["lang"].dropna().astype(str).str.lower()
                if not lang_series.empty:
                    val = lang_series.iloc[0]
                    if val.startswith("en"):
                        lang = "en"
                    elif val.startswith("nl"):
                        lang = "nl"
        except Exception:
            pass

        n_segments = 1
        remove_stopwords = constants.NO

        try:
            tagger = Tagger(
                lang=lang,
                n_segments=n_segments,
                remove_stopwords=remove_stopwords,
                text_tuples=self.stories,
                custom_tags_and_word_column=None,
                callback=None,
                use_infinitives=False,
            )
        except TypeError:
            return None

        return tagger.complete_data


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
        auto_elements = Msg("No Story elements input; generated internally from Stories.")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.actiontagger = ActionTagger(constants.NL_SPACY_MODEL)
        self._nlp = None  # will hold the spaCy model when needed
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

    # === NEW: spaCy helpers for voice/tense ===
    def _ensure_spacy_model(self):
        """
        Lazily load the spaCy model used for voice/tense detection.
        """
        if getattr(self, "_nlp", None) is not None:
            return
        try:
            self._nlp = spacy.load(constants.NL_SPACY_MODEL)
        except Exception:
            # Fallback: try a generic small English model
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None


    @Inputs.stories
    def set_stories(self, stories=None):
        """Stories expects a Corpus. Because Corpus is a subclass of Table, Orange type checking 
        misses wrongly connected inputs.         
        """
        self.valid_stories = []
        if self.story_elements is not None:
            self.Error.clear()
            self.start(
                self.run,
                self.story_elements
            )
            self.postags_box.setEnabled(True)
            self.custom_tags.setChecked(False)
            self.custom_tags.setEnabled(False)
        else:
            # No Story elements input: generate them internally from Stories
            auto_df = self._auto_generate_story_elements()
            if auto_df is not None:
                self.story_elements = auto_df

                if "lang" in self.story_elements.columns:
                    lang_val = (
                        self.story_elements["lang"]
                        .dropna()
                        .astype(str)
                        .iloc[0]
                    )
                else:
                    lang_val = "nl"

                self.actiontagger = ActionTagger(lang_val)
                self.Warning.auto_elements()
                self.start(self.run, self.story_elements)
                self.postags_box.setEnabled(True)
                self.custom_tags.setChecked(False)
                self.custom_tags.setEnabled(False)
            else:
                self.custom_tags.setChecked(False)
                self.custom_tags.setEnabled(False)
                self.postags_box.setEnabled(False)
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

        # === NEW: verb merging helper for better action detection ===
    def _merge_modal_and_main_verbs(self):
        """
        Optionally merge simple modal + main verb combinations in story_elements.
        This function is defensive: if the expected columns are not present,
        it does nothing.
        """
        if self.story_elements is None:
            return

        df = self.story_elements

        required_cols = ["storyid", "sentence", "token_text_lowercase"]
        if not all(c in df.columns for c in required_cols):
            # underlying ActionTagger does its own verb detection
            return

        # If your tagged elements have POS or token index columns, you can extend
        # this method to use them. For now, we implement a simple text-based merge.
        # Example simple approach (pseudo-implementation placeholder):
        # - For each sentence, look for "will <verb>" or "zal <verb>" etc.
        # - Create a new column 'merged_action_candidate' with the phrase.
        # This is demonstration-level; the robust logic stays in ActionTagger.
        pass  # <- keep here if you don't want to modify df structurally


    def _auto_generate_story_elements(self):
        """
        Generate a story elements dataframe by running the Tagger internally
        when no 'Story elements' input is connected.

        Uses the same defaults as the Elements widget:
        - language: 'nl' (tries to infer from a 'lang' column first)
        - number of segments: 1
        - remove stopwords: constants.NO
        """
        if self.stories is None or len(self.stories) == 0:
            return None

        # Default language
        lang = "nl"
        try:
            df, _ = table_to_frames(self.stories)
            if "lang" in df.columns:
                lang_series = df["lang"].dropna().astype(str).str.lower()
                if not lang_series.empty:
                    val = lang_series.iloc[0]
                    if val.startswith("en"):
                        lang = "en"
                    elif val.startswith("nl"):
                        lang = "nl"
        except Exception:
            # Fallback: keep default 'nl'
            pass

        n_segments = 1
        remove_stopwords = constants.NO

        try:
            tagger = Tagger(
                lang=lang,
                n_segments=n_segments,
                remove_stopwords=remove_stopwords,
                text_tuples=self.stories,
                custom_tags_and_word_column=None,
                callback=None,
                use_infinitives=False,
            )
        except TypeError:
            # In case the Tagger signature changes, do nothing instead of crashing
            return None

        # Tagger exposes the story elements table as `complete_data`
        return tagger.complete_data



    def _merge_modal_and_main_verbs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge simple modal + main verb combinations into a single action token,
        such as 'will go' or 'zal gaan'.

        This operates at the story-elements level, before ActionTagger aggregates
        verb frequencies. It is defensive: if the expected columns are missing,
        the original DataFrame is returned unchanged.
        """
        if df is None or df.empty:
            return df

        required_cols = ["storyid", "sentence", "token_text_lowercase", "spacy_tag"]
        if not all(c in df.columns for c in required_cols):
            # Not enough information to perform merging
            return df

        FUTURE_AUX_LEMMAS = {"will", "shall", "zullen", "zal", "gaan"}

        df = df.copy()

        def process_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            cols = list(group.columns)
            token_col_idx = cols.index("token_text_lowercase")
            assoc_idx = cols.index("associated_action") if "associated_action" in cols else None

            # Iterate in order of appearance within the sentence
            for i in range(len(group) - 1):
                row = group.iloc[i]
                row_next = group.iloc[i + 1]

                if (
                    str(row["spacy_tag"]) == "AUX"
                    and str(row["token_text_lowercase"]).lower() in FUTURE_AUX_LEMMAS
                    and str(row_next["spacy_tag"]) == "VERB"
                ):
                    phrase = f"{row['token_text_lowercase']} {row_next['token_text_lowercase']}"

                    # Update token_text_lowercase for both tokens
                    group.iat[i, token_col_idx] = phrase
                    group.iat[i + 1, token_col_idx] = phrase

                    # If an 'associated_action' column exists, keep it in sync
                    if assoc_idx is not None:
                        group.iat[i, assoc_idx] = phrase
                        group.iat[i + 1, assoc_idx] = phrase

            return group

        df = df.groupby(["storyid", "sentence"], group_keys=False).apply(process_group)
        return df

    
    
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
                self._merge_modal_and_main_verbs()
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
        # NEW: sort by highest frequency column before sending
        sort_col = None
        for cand in ["abs_freq_text", "rel_freq_text"]:
            if cand in self.action_results_df.columns:
                sort_col = cand
                break

        if sort_col is not None:
            action_results_sorted = self.action_results_df.sort_values(
                by=sort_col, ascending=False
            )
        else:
            action_results_sorted = self.action_results_df

        self.Outputs.story_collection_results.send(
            table_from_frame(action_results_sorted)
        )
    
    def run(self, story_elements, state: TaskState):
        """
        Long-running computation executed in a separate thread.

        Parameters
        ----------
        story_elements : pandas.DataFrame
            Story elements table converted from the Orange Table.
        state : TaskState
            Orange concurrency state used for reporting progress / cancellation.

        Returns
        -------
        tuple
            (action_results_df, valid_stories, selected_action_table_df,
             full_action_table_df, selected_custom_freq, full_custom_freq)
        """
        def advance(progress: float):
            if state.is_interruption_requested():
                raise InterruptedError
            # progress expected in [0, 100]
            state.set_progress_value(progress)

        # Cache the story elements frame used throughout the widget
        self.story_elements = story_elements

        # Compute the core action statistics using the tagger
        self.action_results_df = self.actiontagger.generate_action_analysis_results(
            self.story_elements,
            callback=advance,
        )

        # Build a mapping from storyid -> sub-dataframe and collect valid stories
        self.valid_stories = []
        self.story_elements_dict = {}

        if self.story_elements is not None:
            grouped = self.story_elements.groupby("storyid")
            for storyid, story_df in grouped:
                # storyid in the elements table is usually a plain index (e.g. "0")
                # while the Stories corpus is indexed numerically. We try to
                # convert to int; if that fails we fall back to handling IDs like "ST0".
                idx = None
                try:
                    idx = int(storyid)
                except (TypeError, ValueError):
                    if isinstance(storyid, str) and storyid.startswith("ST"):
                        try:
                            idx = int(storyid[2:])
                        except ValueError:
                            idx = None
                if idx is not None and self.stories is not None and idx < len(self.stories):
                    self.valid_stories.append(self.stories[idx])

                # Always store story_elements keyed by *string* storyid
                self.story_elements_dict[str(storyid)] = story_df

        # (Re)build the per-action summary tables and custom-tag statistics
        selected_indices = sorted(self.selected_documents) if self.selected_documents else []
        self._rebuild_action_tables_for_selection(selected_indices)

        return (
            self.action_results_df,
            self.valid_stories,
            self.selected_action_table_df,
            self.full_action_table_df,
            self.selected_custom_freq,
            self.full_custom_freq,
        )

    def _rebuild_action_tables_for_selection(self, selected_indices: Iterable[int]) -> None:
        """
        Helper used both in `run` and `update_selected_action_results` to
        construct:
            - `full_action_table_df`: all stories
            - `selected_action_table_df`: only selected stories
            - `selected_action_results_df`: filtered action_results_df
            - custom frequency tables (if custom tags exist)
        """
        if self.story_elements is None or self.story_elements.empty:
            self.full_action_table_df = None
            self.selected_action_table_df = None
            self.selected_action_results_df = None
            self.selected_custom_freq = None
            self.full_custom_freq = None
            return

        # Determine which column should be used for listing entities
        self.word_col = next(
            (col for col in self.story_elements.columns if col.startswith("custom_")),
            None,
        )
        if self.word_col is None:
            self.word_col = "token_text_lowercase"

        # Keep only rows that correspond to actual actions
        if "associated_action_lowercase" in self.story_elements.columns:
            mask_actions = ~self.story_elements["associated_action_lowercase"].astype(str).str.contains(r"\?")
            only_actions_df = self.story_elements[mask_actions].copy()
        else:
            # Fallback: keep rows with verb-like navigator tags
            tag_series = self.story_elements.get("story_navigator_tag")
            if tag_series is not None:
                only_actions_df = self.story_elements[tag_series.astype(str).str.contains("VB")].copy()
            else:
                only_actions_df = self.story_elements.copy()

        # ---- Full action table (all stories) ----
        if not only_actions_df.empty and {"associated_action_lowercase", "story_navigator_tag"}.issubset(only_actions_df.columns):
            full_group = (
                only_actions_df
                .groupby(["associated_action_lowercase", "story_navigator_tag"])[self.word_col]
                .agg(lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v)})))
                .reset_index()
            )
            self.full_action_table_df = full_group.rename(
                columns={
                    "associated_action_lowercase": "action",
                    "story_navigator_tag": "entities_type",
                    self.word_col: "entities",
                }
            )
        else:
            # Safe empty frame with expected columns
            self.full_action_table_df = only_actions_df.iloc[0:0].copy()

        # ---- Selected action table (subset of stories) ----
        selected_storyids_elements = [str(i) for i in selected_indices]
        if selected_storyids_elements:
            selected_only = only_actions_df[
                only_actions_df["storyid"].astype(str).isin(selected_storyids_elements)
            ].copy()
        else:
            selected_only = only_actions_df.iloc[0:0].copy()

        if not selected_only.empty and {"associated_action_lowercase", "story_navigator_tag"}.issubset(selected_only.columns):
            selected_group = (
                selected_only
                .groupby(["associated_action_lowercase", "story_navigator_tag"])[self.word_col]
                .agg(lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v)})))
                .reset_index()
            )
            self.selected_action_table_df = selected_group.rename(
                columns={
                    "associated_action_lowercase": "action",
                    "story_navigator_tag": "entities_type",
                    self.word_col: "entities",
                }
            )
        else:
            self.selected_action_table_df = self.full_action_table_df.iloc[0:0].copy()

        # ---- Selected action results (statistics) ----
        if self.action_results_df is not None and not self.action_results_df.empty:
            if "storyid" in self.action_results_df.columns:
                selected_storyids_results = {"ST" + str(i) for i in selected_indices}
                mask_sel = self.action_results_df["storyid"].astype(str).isin(selected_storyids_results)
                self.selected_action_results_df = self.action_results_df[mask_sel].drop(
                    columns=["storyid"],
                    errors="ignore",
                )
            else:
                self.selected_action_results_df = self.action_results_df.copy()
        else:
            self.selected_action_results_df = None

        # ---- Custom tag frequencies ----
        if util.frame_contains_custom_tag_columns(self.story_elements):
            self.custom_tags.setEnabled(True)
            selected_stories = [str(i) for i in selected_indices] if selected_indices else None
            self.selected_custom_freq = self.actiontagger.calculate_customfreq_table(
                self.story_elements,
                selected_stories=selected_stories,
            )
            self.full_custom_freq = self.actiontagger.calculate_customfreq_table(
                self.story_elements,
                selected_stories=None,
            )
        else:
            self.custom_tags.setChecked(False)
            self.custom_tags.setEnabled(False)
            self.selected_custom_freq = None
            self.full_custom_freq = None

    def _detect_voice_and_tense_for_sentence(self, text: str):
        """
        Return (voice, tense) for a given sentence string.
        voice: "active" or "passive" or "unknown"
        tense: "past", "present", "future", or "unknown"
        """
        if not text:
            return "unknown", "unknown"

        self._ensure_spacy_model()
        if self._nlp is None:
            return "unknown", "unknown"

        doc = self._nlp(text)

        # Passive if we see a passive subject or auxiliary
        is_passive = any(
            token.dep_ in ("nsubjpass", "auxpass")
            for token in doc
        )
        voice = "passive" if is_passive else "active"

        FUTURE_AUX_LEMMAS = {"will", "shall", "zullen", "zal", "gaan"}

        tense = "unknown"
        # explicit future
        if any(
            token.lemma_.lower() in FUTURE_AUX_LEMMAS and token.pos_ == "AUX"
            for token in doc
        ) or any("Tense=Fut" in str(token.morph) for token in doc):
            tense = "future"
        elif any("Tense=Past" in str(token.morph) for token in doc):
            tense = "past"
        elif any("Tense=Pres" in str(token.morph) for token in doc):
            tense = "present"

        return voice, tense


    
    # === NEW: derived frequency metrics for Action results ===
    def _add_frequency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add:
        - abs_freq_text  : absolute frequency per text
        - rel_freq_text  : relative frequency per text
        - nominal_ratio  : subject / total frequency (if subject freq is available)
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        story_col = "storyid" if "storyid" in df.columns else None
        if story_col is None:
            return df

        # Action column: pick a column whose name contains "action"
        entity_col = None
        for c in df.columns:
            if "action" in c.lower() and df[c].dtype == object:
                entity_col = c
                break
        if entity_col is None:
            # fallback to first non-story text column
            candidate_entity_cols = [
                c
                for c in df.columns
                if c not in [story_col, "lang"]
                and df[c].dtype == object
            ]
            if not candidate_entity_cols:
                return df
            entity_col = candidate_entity_cols[0]

        freq_cols = [
            c
            for c in df.columns
            if "freq" in c.lower()
            and np.issubdtype(df[c].dtype, np.number)
        ]
        if not freq_cols:
            return df

        base_freq_col = freq_cols[0]

        subj_col = None
        for c in freq_cols:
            cl = c.lower()
            if "subj" in cl or "subject" in cl:
                subj_col = c
                break

        group_cols = [story_col, entity_col]
        grouped = (
            df.groupby(group_cols)[base_freq_col]
            .sum()
            .reset_index(name="abs_freq_text")
        )
        grouped["total_freq_text"] = grouped.groupby(story_col)["abs_freq_text"].transform("sum")
        grouped["rel_freq_text"] = grouped["abs_freq_text"] / grouped["total_freq_text"].replace(
            0, np.nan
        )

        df = df.merge(
            grouped[group_cols + ["abs_freq_text", "rel_freq_text"]],
            on=group_cols,
            how="left",
        )

        if subj_col is not None:
            df["nominal_ratio"] = np.where(
                df[base_freq_col] > 0,
                df[subj_col] / df[base_freq_col],
                np.nan,
            )

        return df
    

    def _add_voice_and_tense_to_results(self):
        """
        Use story_elements to compute voice/tense per action and merge into
        self.action_results_df as columns 'voice' and 'tense'.
        """
        if self.story_elements is None or self.action_results_df is None:
            return

        df = self.story_elements

        required_cols = ["storyid", "sentence"]
        if not all(c in df.columns for c in required_cols):
            return

        action_col_el = None
        # column for action label in elements frame
        for c in df.columns:
            if "associated_action" in c.lower():
                action_col_el = c
                break
        if action_col_el is None:
            return

        # We avoid pseudo-actions marked with '?'
        df_actions = df[~df[action_col_el].astype(str).str.contains(r"\?")]

        records = []
        for (sid, action_label), group in df_actions.groupby(["storyid", action_col_el]):
            # reuse get_el_story_text to rebuild story string from sentences
            text = self.get_el_story_text(group)
            voice, tense = self._detect_voice_and_tense_for_sentence(text)

            # storyid in action_results_df is usually 'ST0', 'ST1', ...
            sid_str = str(sid)
            if not sid_str.startswith("ST"):
                sid_res = f"ST{sid_str}"
            else:
                sid_res = sid_str

            records.append(
                {
                    "storyid": sid_res,
                    "associated_action_lowercase": action_label,
                    "voice": voice,
                    "tense": tense,
                }
            )

        if not records:
            return

        vt_df = pd.DataFrame.from_records(records)

        # Determine which action column is in the results data frame
        merge_keys = ["storyid", "associated_action_lowercase"]
        if "associated_action_lowercase" not in self.action_results_df.columns:
            if "action" in self.action_results_df.columns:
                vt_df = vt_df.rename(columns={"associated_action_lowercase": "action"})
                merge_keys = ["storyid", "action"]
            else:
                # cannot merge sensibly
                return

        self.action_results_df = self.action_results_df.merge(
            vt_df[merge_keys + ["voice", "tense"]],
            on=merge_keys,
            how="left",
        )


    def _cleanup_legacy_action_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df = df.copy()
        drop_candidates = [
            "Agency",
            "agency",
            "Prominence_sf",
            "prominence_sf",
            "raw_freq",
            "raw_frequency",
            "subject_freq",
            "subject_frequency",
            "subj_freq",
        ]
        drop_cols = [c for c in drop_candidates if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def _prepare_action_results(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_frequency_metrics(df)
        df = self._cleanup_legacy_action_columns(df)
        return df

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
        """
        Refresh the cached dataframes when the document selection changes.
        This keeps:
            - `selected_action_results_df`
            - `selected_action_table_df`
            - custom frequency tables
        in sync with the current selection in the document list.
        """
        if self.action_results_df is None or len(self.action_results_df) == 0:
            return

        selected_indices = sorted(self.selected_documents)
        self._rebuild_action_tables_for_selection(selected_indices)

        # When custom tags are present, keep the outgoing table in sync as well.
        if util.frame_contains_custom_tag_columns(self.story_elements) and self.full_custom_freq is not None:
            self.Outputs.customfreq_table.send(
                table_from_frame(self.full_custom_freq)
            )



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
                            self.story_elements_dict[c_index]
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
    WidgetPreview(OWSNActionAnalysis).run(None)
