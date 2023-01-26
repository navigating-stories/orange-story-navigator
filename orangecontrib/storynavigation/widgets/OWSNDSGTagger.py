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
)
from Orange.data import Variable
from Orange.data.domain import Domain, filter_visible
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, DomainContextHandler
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecanvas.gui.utils import disconnected
from orangewidget.utils.listview import ListViewSearch

from orangecontrib.text.corpus import Corpus
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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


class OWSNDSGTagger(OWWidget, ConcurrentWidgetMixin):
    name = "Word Highlighter"
    description = "Identifies named entities and part-of-speech tokens (nouns, adjectives, verbs etc.) in text"
    icon = "icons/dsgtagger.svg"
    priority = 500

    NL_SPACY_MODEL = "nl_core_news_lg" 

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        matching_docs = Output("Matching Docs", Corpus, default=True)
        other_docs = Output("Other Docs", Corpus)
        corpus = Output("Corpus", Corpus)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    
    tag_type = Setting(1)
    search_features: List[Variable] = ContextSetting([])
    display_features: List[Variable] = ContextSetting([])
    selected_documents: Set[int] = Setting({0}, schema_only=True)
    regexp_filter = ContextSetting("")
    show_tokens = Setting(False)
    autocommit = Setting(True)
    
    # POS
    vbz = Setting(True)
    nouns = Setting(True)
    adj = Setting(True)
    pron = Setting(True)
    adp = Setting(True)
    adv = Setting(True)
    conj = Setting(True)
    det = Setting(True)
    num = Setting(True)
    prt = Setting(True)

    # NER
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
        
    # panels for pos and ner tag selection
    postags_box = None
    nertags_box = None

    class Warning(OWWidget.Warning):
        no_feats_search = Msg("No features included in search.")
        no_feats_display = Msg("No features selected for display.")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.corpus = None  # Corpus
        self.nlp_nl = None
        self.__pending_selected_documents = self.selected_documents

        # Search features
        ex_sel = QListView.ExtendedSelection
        # search_box = gui.widgetBox(self.controlArea, "Search features")
        self.search_listbox = sl = VariableListViewSearch(selectionMode=ex_sel)
        # search_box.layout().addWidget(sl)
        sl.setModel(VisibleDomainModel(separators=False))
        sl.selectionModel().selectionChanged.connect(self.search_features_changed)

        # Display features
        # display_box = gui.widgetBox(self.controlArea, "Display features")
        self.display_listbox = dl = VariableListViewSearch(selectionMode=ex_sel)
        # display_box.layout().addWidget(dl)
        dl.setModel(VisibleDomainModel(separators=False))
        dl.selectionModel().selectionChanged.connect(self.display_features_changed)

        # Tag type selection
        tag_type_panel = gui.widgetBox(self.controlArea, "Category of words to highlight:", orientation=Qt.Horizontal,sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.tagtype_box = box = gui.radioButtonsInBox(self.controlArea, self, "tag_type", [], callback=self._tagtype_changed)
        self.named_entities = gui.appendRadioButton(box, "Named Entities")
        self.pos_tags = gui.appendRadioButton(box, "Parts of Speech")
        tag_type_panel.layout().addWidget(box)

        # POS tag list
        self.postags_box = gui.vBox(self.controlArea, "Parts of Speech to highlight:", sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        gui.checkBox(self.postags_box, self, "vbz", "Verbs",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "nouns", "Nouns",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "adj", "Adjectives",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "adp", "Prepositions / Postpositions",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "adv", "Adverbs",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "conj", "Conjunctives",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "det", "Determinative",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "num", "Numericals",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "prt", "Particles",callback=self.pos_selection_changed)
        gui.checkBox(self.postags_box, self, "pron", "Personal pronouns",callback=self.pos_selection_changed)
        self.controlArea.layout().addWidget(self.postags_box)

        # NER tag list
        self.nertags_box = gui.vBox(self.controlArea, "Named entities to highlight:", sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        gui.checkBox(self.nertags_box, self, "per", "People",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "gpe", "Countries, cities, regions",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "loc", "Other kinds of locations",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "norp", "Nationalities and religious or political groups",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "fac", "Buildings, airports, highways, bridges etc.",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "org", "Companies, agencies, institutions, etc.",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "product", "Objects, vehicles, foods, etc. (Not services)",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "eventner", "Named hurricanes, battles, wars, sports events, etc.",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "work_of_art", "Titles of books, songs, etc.",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "law", "Named documents made into laws",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "language", "Any named language",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "date", "Absolute or relative dates or periods",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "time", "Times smaller than a day",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "percent", "Percentages",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "money", "Monetary values",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "quantity", "Measurements, as of weight or distance",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "ordinal", "'first', 'second', etc.",callback=self.ner_selection_changed)
        gui.checkBox(self.nertags_box, self, "cardinal", "Numerals that do not fall under another category",callback=self.ner_selection_changed)
        
        self.controlArea.layout().addWidget(self.nertags_box)
        self.nertags_box.setEnabled(False)

        # Auto-commit box
        gui.auto_commit(self.controlArea, self, "autocommit", "Send data", "Auto send is on", orientation=Qt.Horizontal,sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        # Search
        self.filter_input = gui.lineEdit(self.mainArea,self,"regexp_filter",orientation=Qt.Horizontal,sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),label="RegExp Filter:",callback=self.refresh_search)

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
        return nlp

    def _tagtype_changed(self):
        if self.tag_type == 1:
            self.postags_box.setEnabled(True)
            self.nertags_box.setEnabled(False)
        else:
            self.postags_box.setEnabled(False)
            self.nertags_box.setEnabled(True)
        
        self.show_docs()
        self.commit.deferred()

    def pos_selection_changed(self):
        self.show_docs()
        self.commit.deferred()

    def ner_selection_changed(self):
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
        # if not self.corpus.has_tokens():
        #     self.show_tokens_checkbox.setCheckState(Qt.Unchecked)
        #     #
        #     # self.tagtype_box.setCheckState(Qt.Unchecked)

        # self.show_tokens_checkbox.setEnabled(self.corpus.has_tokens())
        # #
        # self.tagtype_box.setEnabled(self.corpus.has_tokens())

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
        self.selected_documents = self.get_selected_indexes()
        self.show_docs()
        self.commit.deferred()

    def show_docs(self):
        """Show the selected documents in the right area"""
        if self.corpus is None:
            return

        self.Warning.no_feats_display.clear()
        if len(self.display_features) == 0:
            self.Warning.no_feats_display()

        if self.show_tokens:
            tokens = list(self.corpus.ngrams_iterator(include_postags=True))

        parts = []
        for doc_count, c_index in enumerate(sorted(self.selected_documents)):
            text = ""
            for feature in self.display_features:
                value = str(self.corpus[c_index, feature.name])

                if feature.name == 'content':
                    if (self.tag_type == 1):
                        value = self.__postag_text(value)
                    else:
                        value = self.__nertag_text(value)

                if feature in self.search_features:
                    value = self.__mark_text(value)
                
                if feature.name != 'content':
                    value = value.replace("\n", "<br/>")

                is_image = feature.attributes.get("type", "") == "image"
                if is_image and value != "?":
                    value = os.path.join(feature.attributes.get("origin", ""), value)
                    value = '<img src="{}"></img>'.format(value)
                
                text += (
                    f'<tr><td class="variables"><strong>{feature.name}:</strong></td>'
                    f'<td class="content">{value}</td></tr>'
                )

            if self.show_tokens:
                tokens_ = "".join(
                    f'<span class="token">{token}</span>' for token in tokens[c_index]
                )
                text += (
                    f'<tr><td class="variables"><strong>Tokens & Tags:</strong></td>'
                    f"<td>{tokens_}</td></tr>"
                )
            parts.append(text)

        joined = SEPARATOR.join(parts)
        html = f"<table>{joined}</table>"
        base = QUrl.fromLocalFile(__file__)
        self.doc_webview.setHtml(HTML.format(html), base)

    def __nertag_text(self, text):
        sents = sent_tokenize(text, language='dutch')
        html = ""

        ner_tags = []
        if (self.per):
            ner_tags.append("PERSON")
        if (self.loc):
            ner_tags.append("LOC")
        if (self.gpe):
            ner_tags.append("GPE")
        if (self.norp):
            ner_tags.append("NORP")
        if (self.fac):
            ner_tags.append("FAC")
        if (self.org):
            ner_tags.append("ORG")
        if (self.product):
            ner_tags.append("PRODUCT")
        if (self.eventner):
            ner_tags.append("EVENT")
        if (self.work_of_art):
            ner_tags.append("WORK_OF_ART")
        if (self.law):
            ner_tags.append("LAW")
        if (self.language):
            ner_tags.append("LANGUAGE")
        if (self.date):
            ner_tags.append("DATE")
        if (self.time):
            ner_tags.append("TIME")
        if (self.percent):
            ner_tags.append("PERCENT")
        if (self.money):
            ner_tags.append("MONEY")
        if (self.quantity):
            ner_tags.append("QUANTITY")
        if (self.ordinal):
            ner_tags.append("ORDINAL")
        if (self.cardinal):
            ner_tags.append("CARDINAL")

        options = {"ents" : ner_tags, "colors" : {}}

        for sentence in sents:
            tagged_sentence = self.nlp_nl(sentence)
            html += displacy.render(tagged_sentence, style="ent", options = options)
        
        return html

    def __postag_text(self, text):
        pos_tags = []
        if (self.vbz):
            pos_tags.append("VERB")
        if (self.adj):
            pos_tags.append("ADJ")
        if (self.nouns):
            pos_tags.append("NOUN")
        if (self.pron):
            pos_tags.append("PRON")
        if (self.adp):
            pos_tags.append("ADP")
        if (self.adv):
            pos_tags.append("ADV")
        if (self.conj):
            pos_tags.append("CONJ")
        if (self.det):
            pos_tags.append("DET")
        if (self.num):
            pos_tags.append("NUM")
        if (self.prt):
            pos_tags.append("PRT")
        
        sents = sent_tokenize(text, language='dutch')
        
        html = ""
        for sentence in sents:
            tagged_sentence = self.nlp_nl(sentence)
            tags = []
            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_))
        
            from nltk.tokenize import WhitespaceTokenizer
            spans = list(WhitespaceTokenizer().span_tokenize(sentence))

            ents = []
            for tag, span in zip(tags, spans):
                if tag[1] in pos_tags:
                    if tag[1] == 'PRON':
                        print(tag[2])
                        if tag[2] == 'PPER':
                            ents.append({"start" : span[0], 
                                    "end" : span[1], 
                                    "label" : tag[1] })
                    else:
                        ents.append({"start" : span[0], 
                                    "end" : span[1], 
                                    "label" : tag[1] })

            doc = {"text" : sentence, "ents" : ents}

            colors = {"PRON": "blueviolet",
                "VERB": "lightpink",
                "NOUN": "turquoise",
                "ADJ" : "lime",
                "ADP" : "khaki",
                "ADV" : "orange",
                "CONJ" : "cornflowerblue",
                "DET" : "forestgreen",
                "NUM" : "salmon",
                "PRT" : "yellow"}
        
            options = {"ents" : pos_tags, "colors" : colors}
            html += displacy.render(doc, style = "ent", options = options, manual = True)
        
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
                self.doc_list.setCurrentIndex(self.doc_list.model().index(0, 0))
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
            self.n_tokens = sum(map(len, self.corpus.tokens)) if has_tokens else "n/a"
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
            annotated_corpus = create_annotated_table(self.corpus, selected_docs)
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

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    corpus_ = Corpus.from_file("book-excerpts")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)
    WidgetPreview(OWSNDSGTagger).run(corpus_)