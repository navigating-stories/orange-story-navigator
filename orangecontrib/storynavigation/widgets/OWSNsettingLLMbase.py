from typing import Optional
from AnyQt.QtCore import Signal, Qt
from AnyQt.QtGui import QFocusEvent
from AnyQt.QtWidgets import QTextEdit
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
from Orange.data import Table, StringVariable
from Orange.widgets import gui
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.settings import Setting, DomainContextHandler, ContextSetting
from Orange.widgets.widget import OWWidget, Input, Msg

MODELS = ["sshleifer/distilbart-cnn-12-6"]

class TextEdit(QTextEdit):
    sigEditFinished = Signal()

    def focusOutEvent(self, ev: QFocusEvent):
        self.sigEditFinished.emit()
        super().focusOutEvent(ev)


class OWLocalLLMBase(OWWidget, ConcurrentWidgetMixin, openclass=True):
    settingsHandler = DomainContextHandler()
    model_index = Setting(0)
    text_var = ContextSetting(None)
    prompt_start = Setting("")
    prompt_end = Setting("")
    auto_apply = Setting(False)

    class Inputs:
        data = Input("Data", Table)

    class Warning(OWWidget.Warning):        
        missing_str_var = Msg("Data has no text variables.")

    class Error(OWWidget.Error):
        unknown_error = Msg("An error occurred while creating an answer.\n{}")
        model_load_error = Msg("Failed to load the model.")    
       
    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self._data: Optional[Table] = None
        self.__text_var_model = DomainModel(valid_types=(StringVariable,))
        self.__start_text_edit: QTextEdit = None
        self.__end_text_edit: QTextEdit = None

        # Initialize the tokenizer and model
        #try:
        #    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        #    self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        #    print("Model and tokenizer loaded successfully.")
        
        # Initialize the Hugging Face summarization pipeline
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                                       #model="facebook/bart-large-cnn")
            print("Summarization pipeline loaded successfully.")
            
        except Exception as ex:
            self.summarizer = None
            #self.model = None
            self.Error.model_load_error(ex)
            
        
        self.setup_gui()
        self.test_model()  # Test the model to ensure it's working correctly

    def setup_gui(self):
        box = gui.vBox(self.controlArea, "Model")       
        
        gui.comboBox(box, self, "model_index", label="Model:",
                     orientation=Qt.Horizontal,
                     items=["sshleifer/distilbart-cnn-12-6"], callback=self.commit.deferred)

        gui.comboBox(self.controlArea, self, "text_var", "Data",
                     "Text variable:", model=self.__text_var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Prompt")
        
        gui.label(box, self, "Start:")
        self.__start_text_edit = TextEdit(tabChangesFocus=True)
        self.__start_text_edit.setText(self.prompt_start)
        self.__start_text_edit.sigEditFinished.connect(
            self.__on_start_text_edit_changed)
        box.layout().addWidget(self.__start_text_edit)
        
        gui.label(box, self, "End:")
        self.__end_text_edit = TextEdit(tabChangesFocus=True)
        self.__end_text_edit.setText(self.prompt_end)
        self.__end_text_edit.sigEditFinished.connect(
            self.__on_end_text_edit_changed)
        box.layout().addWidget(self.__end_text_edit)

        gui.rubber(self.controlArea)

        gui.auto_apply(self.buttonsArea, self, "auto_apply")

    def __on_start_text_edit_changed(self):
        prompt_start = self.__start_text_edit.toPlainText()
        if self.prompt_start != prompt_start:
            self.prompt_start = prompt_start
            self.commit.deferred()

    def __on_end_text_edit_changed(self):
        prompt_end = self.__end_text_edit.toPlainText()
        if self.prompt_end != prompt_end:
            self.prompt_end = prompt_end
            self.commit.deferred()

    @Inputs.data
    def set_data(self, data: Table):
        self.closeContext()
        self.clear_messages()
        self._data = data
        self.__text_var_model.set_domain(data.domain if data else None)
        self.text_var = self.__text_var_model[0] if self.__text_var_model else None
        if data and not self.__text_var_model:
            self.Warning.missing_str_var()
        self.openContext(data)        
        
    @gui.deferred
    def commit(self):
        self.Error.unknown_error.clear()
        self.start(self.ask_llm)
    
    def ask_llm(self):
        raise NotImplementedError()
        
    def run_summarizer_model(self, text: str) -> str:
        summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    
    def on_exception(self, ex: Exception):
        self.Error.unknown_error(ex)

    def on_partial_result(self, _):
        pass

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()
        
    def test_model(self):
        test_text = "This is a test."
        try:
            result = self.run_summarizer_model(test_text)
            print(f"Test result: {result}")
        except Exception as ex:
            print(f"Error during test: {ex}")
