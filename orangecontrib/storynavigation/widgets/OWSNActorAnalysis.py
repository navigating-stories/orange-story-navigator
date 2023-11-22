import dhtmlparser3
# from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re
import sre_constants
from typing import Any, Iterable, List, Set
import numpy as np

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
    QLabel,
)
from Orange.data import Variable, Table
from Orange.data.domain import Domain, filter_visible
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, DomainContextHandler
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecanvas.gui.utils import disconnected
from orangewidget.utils.listview import ListViewSearch
from Orange.data.pandas_compat import table_from_frame

from orangecontrib.text.corpus import Corpus
from operator import itemgetter

import pandas as pd
import json
import spacy
from spacy import displacy
# import nltk
import matplotlib.pyplot as plt
import numpy as np
# nltk.download('punkt')
# nltk.download('perluniprops')

# import neuralcoref

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
                    QItemSelection(model.index(index, 0),
                                   model.index(index, 0)),
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


class OWSNActorAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = "1) Actor Analysis"
    description = "Provides tools to support basic narrative analysis for actors in stories."
    icon = "icons/actor_analysis_icon.png"
    priority = 6422

    NL_SPACY_MODEL = "nl_core_news_lg"

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        matching_docs = Output("Matching Docs", Corpus, default=True)
        other_docs = Output("Other Docs", Corpus)
        corpus = Output("Corpus", Corpus)
        agency_table = Output("Actor agency ratios", Table)
        halliday_actions_table = Output("Halliday action counts", Table)
        actor_action_table = Output("Actor action table", Table)
        

    settingsHandler = DomainContextHandler()
    settings_version = 2
    search_features: List[Variable] = ContextSetting([])
    display_features: List[Variable] = ContextSetting([])
    selected_documents: Set[int] = Setting({0}, schema_only=True)
    regexp_filter = ContextSetting("")
    show_tokens = Setting(False)
    autocommit = Setting(True)

    # Scoring related to agent prominence score
    agent_prominence_score_max = 0.
    agent_prominence_score_min = 0.
    agent_prominence_metrics = ['Subject frequency', 'Subject frequency (normalized)']
    agent_prominence_metric = 'Subject frequency'

    # Index of word prominence scores for each word in story
    word_prominence_scores = {}

    # HTML string rendering of story document
    html_result = ''

    # POS or NER? radiobutton selection of entity type to highlight
    tag_type = Setting(1)

    # Parts of speech (POS) checkbox selected initialization
    vbz = Setting(True)
    subjs = Setting(True)
    nouns = Setting(True)
    adj = Setting(True)
    pron = Setting(True)
    adp = Setting(True)
    adv = Setting(True)
    conj = Setting(True)
    det = Setting(True)
    num = Setting(True)
    prt = Setting(True)
    propn = Setting(True)
    all_pos = Setting(True)
    zero_pos = Setting(False)

    # Named entity recognition (NER) types checkbox selected initialization
    per = Setting(True)
    loc = Setting(True)
    gpe = Setting(True)
    norp = Setting(True)
    fac = Setting(True)
    org = Setting(True)
    product = Setting(True)
    eventner = Setting(True)
    work_of_art = Setting(True)
    law = Setting(True)
    language = Setting(True)
    date = Setting(True)
    time = Setting(True)
    percent = Setting(True)
    money = Setting(True)
    quantity = Setting(True)
    ordinal = Setting(True)
    cardinal = Setting(True)

    # Panels for POS and NER tag types or lists
    postags_box = None
    nertags_box = None
    main_agents_box = None

    # list of Dutch stopwords
    nl_stopwords = []

    # POS counts initialisation
    noun_count = 0
    verb_count = 0
    adjective_count = 0

    # Other counts initialisation
    word_count = 0
    word_count_nostops = 0
    sentence_count = 0
    sentence_count_per_word = {}
    count_per_word = {}
    count_per_subject = {}
    # count_per_word_passive = {}
    # count_per_word_active = {}
    noun_action_dict = {}

    # original text (not tagged)
    original_text = ''

    sli = None

    # list of colour values for the background highlight for each entity type
    highlight_colors = {}

    # list of punctuation characters
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''

    # list of POS checkboxes for each POS type
    pos_checkboxes = []

    class Warning(OWWidget.Warning):
        no_feats_search = Msg("No features included in search.")
        no_feats_display = Msg("No features selected for display.")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        # loads list of Dutch stopwords
        with open('orangecontrib/storynavigation/utils/dutchstopwords.txt', 'r', encoding='utf8') as f:
            self.nl_stopwords = [line.rstrip() for line in f]

        self.corpus = None  # initialise list of documents (corpus)
        self.nlp_nl = None  # initialise spacy model
        self.__pending_selected_documents = self.selected_documents

        # Search features
        ex_sel = QListView.ExtendedSelection
        self.search_listbox = sl = VariableListViewSearch(selectionMode=ex_sel)
        sl.setModel(VisibleDomainModel(separators=False))
        sl.selectionModel().selectionChanged.connect(self.search_features_changed)

        # Display features
        self.display_listbox = dl = VariableListViewSearch(
            selectionMode=ex_sel)
        dl.setModel(VisibleDomainModel(separators=False))
        dl.selectionModel().selectionChanged.connect(self.display_features_changed)

        # Tag type selection panel
        tag_type_panel = gui.widgetBox(self.controlArea, "Category of words to highlight:",
                                       orientation=Qt.Horizontal, sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.tagtype_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "tag_type", [], callback=self._tagtype_changed)
        self.named_entities = gui.appendRadioButton(box, "Named Entities")
        self.pos_tags = gui.appendRadioButton(box, "Parts of Speech")
        tag_type_panel.layout().addWidget(box)

        # POS tag list
        self.postags_box = gui.vBox(self.controlArea, "Parts of Speech to highlight:", sizePolicy=QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.sc = gui.checkBox(self.postags_box, self, "subjs",
                               "Subjects", callback=self.pos_selection_changed)
        self.vc = gui.checkBox(self.postags_box, self, "vbz",
                               "Actions", callback=self.pos_selection_changed)
        self.nc = gui.checkBox(self.postags_box, self, "nouns",
                               "Entities", callback=self.pos_selection_changed)
        self.adjc = gui.checkBox(
            self.postags_box, self, "adj", "Descriptives", callback=self.pos_selection_changed)
        self.allc = gui.checkBox(self.postags_box, self, "all_pos", "All")
        self.allc.setChecked(False)
        self.allc.stateChanged.connect(self.on_state_changed_pos)

        self.pos_checkboxes = [self.sc, self.vc, self.nc, self.adjc]
        self.controlArea.layout().addWidget(self.postags_box)

        # NER tag list
        self.nertags_box = gui.vBox(self.controlArea, "Named entities to highlight:", sizePolicy=QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        gui.checkBox(self.nertags_box, self, "per", "People",
                     callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "loc", "Places",
                     callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "product",
                     "Other entities", callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "date", "Temporals",
                     callback=self.ner_selection_changed)

        self.controlArea.layout().addWidget(self.nertags_box)
        self.nertags_box.setEnabled(False)

        # Prominence score slider
        self.main_agents_box = gui.vBox(self.controlArea, "Filter entities by prominence score:", sizePolicy=QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.metric_name_combo = gui.comboBox(self.main_agents_box, self, 'agent_prominence_metric',
                                              items=self.agent_prominence_metrics,
                                              sendSelectedValue=True,
                                              callback=self.prominence_metric_change)

        self.main_agents_box.setEnabled(True)

        gui.hSlider(
            self.main_agents_box,
            self,
            "agent_prominence_score_min",
            minValue=0.,
            maxValue=100.,
            # step=.01,
            ticks=True,
            callback=self.slider_callback,
            label='Min:',
            labelFormat="%.1f",
            intOnly=False,
        )

        self.controlArea.layout().addWidget(self.main_agents_box)

        gui.spin(
            self.main_agents_box, self, "agent_prominence_score_min", minv=0., maxv=100.,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.slider_callback)

        # Auto-commit box
        gui.auto_commit(self.controlArea, self, "autocommit", "Send data", "Auto send is on",
                        orientation=Qt.Horizontal, sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        # Search
        self.filter_input = gui.lineEdit(self.mainArea, self, "regexp_filter", orientation=Qt.Horizontal, sizePolicy=QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed), label="RegExp Filter:", callback=self.refresh_search)

        # Main area
        self.splitter = QSplitter(
            orientation=Qt.Horizontal, childrenCollapsible=False)
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
        # self.doc_webview.setStyleSheet("QWidget {background-color:   #0ff}")
        self.mainArea.layout().addWidget(self.splitter)

    def on_state_changed_pos(self, state):
        for checkBox in self.pos_checkboxes:
            checkBox.setCheckState(state)

    def copy_to_clipboard(self):
        text = self.doc_webview.selectedText()
        QApplication.clipboard().setText(text)

    def load_spacy_pipeline(self, name):
        """Check if the spacy language pipeline was downloaded and load it.
        Downloads the language pipeline if not available.

        Args:
            name (string): Name of the spacy language.

        Returns:
            spacy.language.Language: The spacy language pipeline
        """
        if spacy.util.is_package(name):
            nlp = spacy.load(name)
        else:
            os.system(f"spacy download {name}")
            nlp = spacy.load(name)
            nlp.add_pipe("merge_noun_chunks")
            nlp.add_pipe("merge_entities")
        return nlp

    def _tagtype_changed(self):
        """ Toggles the disabling and enabling of the list of checkboxes associated with 
        Parts of Speech vs. Named Entities. The user cannot highlight entities of both
         categories. When POS tags are selected, the list of NER checkboxes are disabled
        and vice versa. This function takes care of this.
        """
        if self.tag_type == 1:
            self.postags_box.setEnabled(True)
            self.main_agents_box.setEnabled(True)
            self.nertags_box.setEnabled(False)
        else:
            self.postags_box.setEnabled(False)
            self.main_agents_box.setEnabled(False)
            self.nertags_box.setEnabled(True)

        self.show_docs()
        self.commit.deferred()

    def pos_selection_changed(self):
        self.show_docs()
        self.commit.deferred()

    def ner_selection_changed(self):
        self.show_docs()
        self.commit.deferred()

    def rehighlight_entities(self):
        self.show_docs()
        self.commit.deferred()

    @Inputs.corpus
    def set_data(self, corpus=None):
        self.nlp_nl = self.load_spacy_pipeline(self.NL_SPACY_MODEL)

        self.closeContext()
        self.reset_widget()
        self.corpus = corpus
        if corpus is not None:
            self.setup_controls()
            self.openContext(self.corpus)
            self.doc_list.model().set_filter_string(self.regexp_filter)
            self.select_variables()
            self.list_docs()
            self.update_info()
            self.set_selection()
            self.show_docs()
        self.commit.now()

    def reset_widget(self):
        # Corpus
        self.corpus = None
        self.tagtype_box = None
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
        domain = self.corpus.domain

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
        self.display_features = [
            f for f in self.display_features if f in dmodel]
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
        docs = self.regenerate_docs()
        self.doc_list_model.setup_data(self.corpus.titles.tolist(), docs)

    def get_selected_indexes(self) -> Set[int]:
        m = self.doc_list.model().mapToSource
        return {m(i).row() for i in self.doc_list.selectionModel().selectedRows()}

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
            r for r in self.selected_documents if r < len(self.corpus)
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

    def selection_changed(self) -> None:
        """Function is called every time the selection changes"""

        self.agent_prominence_score_min = 0.
        self.word_prominence_scores = {}
        self.html_result = ''

        self.selected_documents = self.get_selected_indexes()
        self.show_docs()
        self.commit.deferred()

    def filter_entities(self):
        if ((len(self.word_prominence_scores) == 0) or (self.html_result == '')):
            self.show_docs(slider_engaged=False)
            self.commit.deferred()
        else:
            entities = set()
            for item in self.word_prominence_scores:
                if self.word_prominence_scores[item] >= self.agent_prominence_score_min:
                    entities.add(item)

            dom = dhtmlparser3.parse(self.html_result)
            for mark in dom.find("mark"):
                current_mark_style = mark.parameters["style"]
                current_mark_bgcolor_str = current_mark_style.split(';')[0]
                current_mark_bgcolor = current_mark_bgcolor_str.split(':')[
                    1].strip()
                newbg_parts = ['background: white']
                newbg_parts.extend(current_mark_style.split(';')[1:])
                whitebg_current_mark_style = ';'.join(newbg_parts)

                stri = mark.content_without_tags().strip()
                stri = re.sub(r'\s+', '.', stri)
                parts = stri.split('.')
                self.highlight_colors[parts[0]] = current_mark_bgcolor

                if parts[0] not in entities:
                    old_params = mark.parameters
                    old_params['style'] = whitebg_current_mark_style
                    mark.replace_with(dhtmlparser3.Tag(
                        "mark", old_params, [parts[0]], is_non_pair=False))
                else:
                    old_params = mark.parameters
                    old_params['style'] = whitebg_current_mark_style.replace(
                        "white", self.highlight_colors[parts[0]])
                    mark.replace_with(dhtmlparser3.Tag(
                        "mark", old_params, [parts[0]], is_non_pair=False))

        return str(dom)

    def calculate_prominence_score(self, word, list_of_sentences, tags):
        score = 0
        word_count = 0
        for sent in list_of_sentences:
            items = sent.split()
            word_count+= len(items)

        if (self.agent_prominence_metric == "Subject frequency (normalized)"):
            score = (1 - ((self.count_per_word[word] - self.count_per_subject[word]) / self.count_per_word[word])) * (self.count_per_word[word] / word_count) * 100
        elif (self.agent_prominence_metric == "Subject frequency"):
            score = self.count_per_subject[word]
        else:
            score = 0

        return score

    def get_max_prominence_score(self):
        highest_score = 0
        for item in self.word_prominence_scores:
            if self.word_prominence_scores[item] > highest_score:
                highest_score = self.word_prominence_scores[item]
        return highest_score

    def prominence_metric_change(self):
        self.agent_prominence_score_min = 0.
        self.word_prominence_scores = {}
        self.html_result = ''
        self.show_docs(slider_engaged=False)
        self.commit.deferred()

    def slider_callback(self):
        if self.agent_prominence_score_min > self.agent_prominence_score_max:
            self.agent_prominence_score_min = self.agent_prominence_score_max
        self.show_docs(slider_engaged=True)
        self.commit.deferred()

    def show_docs(self, slider_engaged=False):
        """Show the selected documents in the right area"""
        if self.corpus is None:
            return

        self.Warning.no_feats_display.clear()
        if len(self.display_features) == 0:
            self.Warning.no_feats_display()

        parts = []
        for doc_count, c_index in enumerate(sorted(self.selected_documents)):
            text = ""
            for feature in self.display_features:
                value = str(self.corpus[c_index, feature.name])
                self.original_text = str(value)

                if feature.name == 'content':
                    if (self.tag_type == 1):
                        if (slider_engaged):
                            value = self.filter_entities()
                        else:
                            value = self.__postag_text(value)
                            self.Outputs.agency_table.send(table_from_frame(self.calculate_agency_table()))
                            self.Outputs.actor_action_table.send(table_from_frame(self.generate_noun_action_table()))
                            self.Outputs.halliday_actions_table.send(table_from_frame(self.generate_halliday_action_counts_table(text=self.original_text)))
                    else:
                        value = self.__nertag_text(value)

                if feature in self.search_features and (len(self.regexp_filter) > 0):
                    value = self.__mark_text(self.original_text)

                if feature.name != 'content':
                    value = value.replace("\n", "<br/>")

                is_image = feature.attributes.get("type", "") == "image"

                if is_image and value != "?":
                    value = os.path.join(
                        feature.attributes.get("origin", ""), value)
                    value = '<img src="{}"></img>'.format(value)

                text += (
                    f'<tr><td class="variables"><strong>{feature.name}:</strong></td>'
                    f'<td class="content">{value}</td></tr>'
                )

            parts.append(text)

        joined = SEPARATOR.join(parts)
        html = f"<table>{joined}</table>"
        base = QUrl.fromLocalFile(__file__)
        self.doc_webview.setHtml(HTML.format(html), base)

    def calculate_agency_table(self):
        rows = []
        n = 10
        res = dict(sorted(self.count_per_subject.items(),
                   key=itemgetter(1), reverse=True)[:n])
        words = list(res.keys())
        subj_count_values = list(res.values())

        for word in words:
            rows.append([word, self.count_per_subject[word], self.count_per_word[word]])
        return pd.DataFrame(rows, columns=['actor', 'subject_frequency', 'raw_frequency'])

    def generate_noun_action_table(self):
        n = 10
        res = dict(sorted(self.word_prominence_scores.items(),
                   key=itemgetter(1), reverse=True)[:n])
        
        names = list(res.keys())

        rows = []
        for item in self.noun_action_dict:
            if len(self.noun_action_dict[item]) > 0 and (item in names):
                curr_row = []
                curr_row.append(item)
                curr_row.append(', '.join(list(set(self.noun_action_dict[item]))))
                rows.append(curr_row)

        return pd.DataFrame(rows, columns=['actor', 'actions'])
    
    def as_list(self, dw):
        res = []
        for item in dw:
            if (type(item) == str and item != 'N/A'):
                words = item.split(' | ')
                if len(words) > 0:
                    for word in words:
                        res.append(word.lower().strip())

        return list(set(res))
    
    def generate_halliday_action_counts_table(self, text, dim_type='realm'):
        rows = []
        
        # Valid values for 'dim_type' parameter: realm, process, prosub, sub
        with open('orangecontrib/storynavigation/utils/halliday_dimensions_' + dim_type + '.json') as json_file:
            halliday_dict = json.load(json_file)

        # Calculate the number of story words in each halliday dimension
        words = text.split()
        halliday_counts = {}
        for item in halliday_dict:
            halliday_counts[item] = 0

        for word in words:
            processed_word = word.lower().strip()
            for item in halliday_dict:
                if processed_word in halliday_dict[item]:
                    halliday_counts[item] += 1

        # Prepare data for the sectors and their values in the polar area chart
        categories = list(halliday_counts.keys())
        values = list(halliday_counts.values())

        for item in halliday_dict:
            rows.append([item, halliday_counts[item]])

        return pd.DataFrame(rows, columns=['action', 'frequency'])

        # # Create the polar area chart
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.set_theta_direction(-1)  # Rotate the plot clockwise
        # ax.set_theta_zero_location('N')  # Set the zero angle at the north

        # # Plot the sectors
        # ax.bar(
        #     [i * (2 * np.pi / len(categories)) for i in range(len(categories))],
        #     values,
        #     width=(2 * np.pi / len(categories)),
        #     align='edge',
        #     color='skyblue',
        # )

        # # Set the sector labels
        # ax.set_xticks([i * (2 * np.pi / len(categories)) for i in range(len(categories))])
        # ax.set_xticklabels(categories)

        # # Display the plot
        # plt.show()

    def calculate_word_type_count(self, sent_models):
        for sent_model in sent_models:
            tags = []
            subjs = []
            
            for token in sent_model:
                if (token.text.lower().strip() not in self.nl_stopwords):
                    if (token.dep_ == 'nsubj' and token.pos_ in ['PRON', 'NOUN', 'PROPN']) or token.text.lower().strip() == 'ik':
                        subjs.append((token, token.idx, token.idx+len(token)))
                    else:
                        self.count_per_word[token.text.lower().strip()] += 1

            subjs = self.sort_tuple(subjs)
            main_subject = ''
            
            if len(subjs) > 0:
                main_subject = subjs[0][0].text
                self.count_per_subject[main_subject.lower().strip()] += 1
                self.count_per_word[main_subject.lower().strip()] += 1

    def __nertag_text(self, text):
        from spacy.lang.nl import Dutch
        nlp = Dutch()
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sents = list(doc.sents)
        # sents = sent_tokenize(text, language='dutch')
        html = ""

        ner_tags = []
        if (self.per):
            ner_tags.append("PERSON")
        if (self.loc):
            ner_tags.append("LOC")
            ner_tags.append("GPE")
            ner_tags.append("NORP")
            ner_tags.append("FAC")
            ner_tags.append("ORG")
        if (self.product):
            ner_tags.append("ORG")
            ner_tags.append("PRODUCT")
            ner_tags.append("EVENT")
            ner_tags.append("WORK_OF_ART")
        if (self.date):
            ner_tags.append("DATE")
            ner_tags.append("TIME")

        options = {"ents": ner_tags, "colors": {}}

        for sentence in sents:
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("  ", " ")
            sentence = re.sub('\s+', ' ', sentence)
            tagged_sentence = self.nlp_nl(sentence)
            html += displacy.render(tagged_sentence,
                                    style="ent", options=options)

        return html

    # Function to recursively traverse ancestors
    def find_verb_ancestor(self, token):
        # Check if the token is a verb
        if token.pos_ == 'VERB':
            return token

        # Traverse the token's ancestors recursively
        for ancestor in token.ancestors:
            # Recursive call to find the verb ancestor
            verb_ancestor = self.find_verb_ancestor(ancestor)
            if verb_ancestor:
                return verb_ancestor

        # If no verb ancestor found, return None
        return None

    def merge_punct(self, doc):
        spans = []
        for word in doc[:-1]:
            if word.is_punct or not word.nbor(1).is_punct:
                continue
            start = word.i
            end = word.i + 1
            while end < len(doc) and doc[end].is_punct:
                end += 1
            span = doc[start:end]
            spans.append((span, word.tag_, word.lemma_, word.ent_type_))
        with doc.retokenize() as retokenizer:
            for span, tag, lemma, ent_type in spans:
                attrs = {"tag": tag, "lemma": lemma, "ent_type": ent_type}
                retokenizer.merge(span, attrs=attrs)
        return doc

    def sort_tuple(self, tup):
        lst = len(tup)
        for i in range(0, lst):
            for j in range(0, lst-i-1):
                if (tup[j][1] > tup[j + 1][1]):
                    temp = tup[j]
                    tup[j] = tup[j + 1]
                    tup[j + 1] = temp
        return tup

    def __postag_text(self, text):
        # pos tags that the user wants to highlight
        pos_tags = []

        # add pos tags to highlight according to whether the user has selected them or not
        if (self.vbz):
            pos_tags.append("VERB")
        if (self.adj):
            pos_tags.append("ADJ")
            pos_tags.append("ADV")
        if (self.nouns):
            pos_tags.append("NOUN")
            pos_tags.append("PRON")
            pos_tags.append("PROPN")
        if (self.subjs):
            pos_tags.append("SUBJ")

        # tokenize input text into sentences
        from spacy.lang.nl import Dutch
        nlp = Dutch()
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sents_spans = list(doc.sents)
        sents = []
        for span in sents_spans:
            sents.append(span.text)

        # sents = sent_tokenize(text, language='dutch')

        # count no. of sents
        sentence_count = len(sents)

        # count no. of words and words not considering stopwords
        from spacy.lang.nl import Dutch
        nlp = Dutch()
        tokenizer = nlp.tokenizer

        for i in range(0, len(sents)):
            sents[i] = sents[i].replace('.', '')

        for sentence in sents:
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("  ", " ")
            sentence = re.sub('\s+', ' ', sentence)
            sentence = sentence.replace('.', '')

            # tokens = word_tokenize(sentence, language='dutch')
            tokens_doc = tokenizer(sentence)
            tokens = []
            for d in tokens_doc:
                tokens.append(d.text)

            self.word_count += len(tokens)
            for token in tokens:
                self.sentence_count_per_word[token.lower().strip()] = 0
                self.count_per_word[token.lower().strip()] = 0
                self.count_per_subject[token.lower().strip()] = 0
                # self.count_per_word_active[token.lower().strip()] = 0
                # self.count_per_word_passive[token.lower().strip()] = 0
                self.noun_action_dict[token.lower().strip()] = []
                if token.lower().strip() not in self.nl_stopwords:
                    self.word_count_nostops += 1

        # count no. of sents that each word appears in
        for sentence in sents:
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("  ", " ")
            sentence = re.sub('\s+', ' ', sentence)
            sentence = sentence.replace('.', '')
            # tokens = word_tokenize(sentence, language='dutch')
            tokens_doc = tokenizer(sentence)
            tokens = []
            for d in tokens_doc:
                tokens.append(d.text)

            for token in tokens:
                self.sentence_count_per_word[token.lower().strip()] += 1

        # output of this function
        html = ""

        # generate and store nlp tagged models for each sentence
        sentence_nlp_models = []
        for sentence in sents:
            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("  ", " ")
            sentence = re.sub('\s+', ' ', sentence)
            sentence = sentence.replace('.', '')

            tagged_sentence = self.nlp_nl(sentence)
            for token in tagged_sentence:
                self.count_per_word[token.text.lower().strip()] = 0
                self.count_per_subject[token.text.lower().strip()] = 0
            # tagged_sentence = self.merge_punct(tagged_sentence)
            sentence_nlp_models.append(tagged_sentence)

        # calculate the number of unique nouns in the text
        self.calculate_word_type_count(sentence_nlp_models)

        # loop through model to filter out those words that need to be tagged (based on user selection and prominence score)
        for sentence, tagged_sentence in zip(sents, sentence_nlp_models):
            tags = []
            subjs = []
            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_, token.dep_))
                if (token.dep_ == 'nsubj' and token.pos_ in ['PRON', 'NOUN', 'PROPN']) or token.text.lower().strip() == 'ik':
                    subjs.append((token, token.idx, token.idx+len(token)))

            subjs = self.sort_tuple(subjs)
            main_subject = ''
            
            if len(subjs) > 0:
                main_subject = subjs[0][0].text
            # for ent in tagged_sentence.ents:
            #     print(ent, " : ", ent._.coref_cluster)

            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
            spans = list(tokenizer.span_tokenize(sentence))

            ents = []
            for tag, span in zip(tags, spans):

                if tag[0].lower().strip() not in self.nl_stopwords:
                    if tag[1] == 'PRON':
                        if ('|' in tag[2]):
                            tmp_tags = tag[2].split('|')
                            if (tmp_tags[1] == 'pers' and tmp_tags[2] == 'pron') or (tag[0].lower().strip() == 'ik'):
                                p_score = 0
                                p_score = self.calculate_prominence_score(
                                    tag[0].lower().strip(), sents, tags)
                                self.word_prominence_scores[tag[0].lower(
                                ).strip()] = p_score

                                if (p_score >= self.agent_prominence_score_min):
                                    if tag[0].lower().strip() == main_subject.lower().strip():
                                        ents.append({"start": span[0],
                                                     "end": span[1],
                                                     "label": "SUBJ"})
                                    else:
                                        ents.append({"start": span[0],
                                                     "end": span[1],
                                                     "label": tag[1]})

                                vb = self.find_verb_ancestor(token)
                                if vb is not None:
                                    self.noun_action_dict[tag[0].lower().strip()].append(
                                        vb.text)

                    elif ((tag[1] == 'NOUN') or (tag[1] == 'PROPN')):
                        p_score = 0
                        p_score = self.calculate_prominence_score(
                            tag[0].lower().strip(), sents, tags)
                        self.word_prominence_scores[tag[0].lower(
                        ).strip()] = p_score

                        if (p_score >= self.agent_prominence_score_min):
                            if tag[0].lower().strip() == main_subject.lower().strip():
                                ents.append({"start": span[0],
                                             "end": span[1],
                                             "label": "SUBJ"})
                            else:
                                ents.append({"start": span[0],
                                             "end": span[1],
                                             "label": tag[1]})

                        vb = self.find_verb_ancestor(token)
                        if vb is not None:
                            self.noun_action_dict[tag[0].lower().strip()].append(
                                vb.text)

                    else:
                        ents.append({"start": span[0],
                                    "end": span[1],
                                     "label": tag[1]})

            # specify sentences and filtered entities to tag / highlight
            doc = {"text": sentence, "ents": ents}

            # specify colors for highlighting each entity type
            colors = {}
            if self.nouns:
                colors["NOUN"] = "turquoise"
                colors["PRON"] = "#BB4CBA"
                colors["PROPN"] = "#259100"
            if self.subjs:
                colors["SUBJ"] = "#FFEB26"
            if self.vbz:
                colors["VERB"] = "lightpink"
            if self.adj:
                colors["ADJ"] = "lime"
                colors["ADP"] = "khaki"
                colors["ADV"] = "orange"

            self.agent_prominence_score_max = self.get_max_prominence_score()
            # collect the above config params together
            options = {"ents": pos_tags, "colors": colors}
            # give all the params to displacy to generate HTML code of the text with highlighted tags
            html += displacy.render(doc, style="ent",
                                    options=options, manual=True)

        self.html_result = html

        return html

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
            text[m.start(): m.end()] = list(
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
        if self.corpus:
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
        return self.corpus.documents_from_features(self.search_features)

    def refresh_search(self):
        if self.corpus is not None:
            self.doc_list.model().set_filter_string(self.regexp_filter)
            if not self.selected_documents:
                # when currently selected items are filtered selection is empty
                # select first element in the view in that case
                self.doc_list.setCurrentIndex(
                    self.doc_list.model().index(0, 0))
            self.update_info()
            self.start(
                _count_matches,
                self.doc_list_model.get_filter_content(),
                self.regexp_filter,
            )
            self.show_docs()
            self.commit.deferred()

    def on_done(self, res: int):
        """When matches count is done show the result in the label"""
        self.n_matches = res if res is not None else "n/a"

    def on_exception(self, ex):
        raise ex

    def update_info(self):
        if self.corpus is not None:
            has_tokens = self.corpus.has_tokens()
            self.n_matching = f"{self.doc_list.model().rowCount()}/{len(self.corpus)}"
            self.n_tokens = sum(map(len, self.corpus.tokens)
                                ) if has_tokens else "n/a"
            self.n_types = len(self.corpus.dictionary) if has_tokens else "n/a"
        else:
            self.n_matching = "n/a"
            self.n_matches = "n/a"
            self.n_tokens = "n/a"
            self.n_types = "n/a"

    @gui.deferred
    def commit(self):
        matched = unmatched = annotated_corpus = None
        if self.corpus is not None:
            selected_docs = sorted(self.get_selected_indexes())
            matched = self.corpus[selected_docs] if selected_docs else None
            mask = np.ones(len(self.corpus), bool)
            mask[selected_docs] = 0
            unmatched = self.corpus[mask] if mask.any() else None
            annotated_corpus = create_annotated_table(
                self.corpus, selected_docs)
        self.Outputs.matching_docs.send(matched)
        self.Outputs.other_docs.send(unmatched)
        self.Outputs.corpus.send(annotated_corpus)

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
                display_features = [f_order[i]
                                    for i in display_idx if i < len(f_order)]
                search_features = [f_order[i]
                                   for i in search_ids if i < len(f_order)]
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

    from orangecontrib.text.preprocess import BASE_TOKENIZER
    

    corpus_ = Corpus.from_file("book-excerpts")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)
    WidgetPreview(OWSNActorAnalysis).run(corpus_)
