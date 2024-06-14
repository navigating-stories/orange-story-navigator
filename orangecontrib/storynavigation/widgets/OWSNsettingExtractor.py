"""
This code constructs a new text field using a Local LLM such has Ollama. It processes multiple texts
from the input data, generates responses for each,and adds a new column to the data table with these
responses. It build upon the base functionality provided by OWLocalLLMBase. 
- Processes each text entry in the input data separately.
- Caches responses to avoid redundant processing for the same inputs.
"""

from typing import List
from Orange.data import Table, StringVariable
from Orange.data.util import get_unique_names
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output
from storynavigation.widgets.OWSNsettingLLMbase import OWLocalLLMBase, MODELS


class OWStorySetting(OWLocalLLMBase):
    name = "StorySetting Extractor"
    description = "Construct a text field with a description of the setting using a Local LLM."
    icon = "icons/storynavigator_logo.png"
    priority = 11
    keywords = ["text", "gpt"]

    cache = Setting({})
    want_main_area = False

    class Outputs:
        data = Output("Data", Table)
    
    def set_data(self, data: Table):
            super().set_data(data)
            self.commit.deferred()
            
    def on_done(self, answers: List[str]):
        data = self._data
        if len(answers) > 0:
            name = get_unique_names(data.domain, "Text")
            var = StringVariable(name)
            data = data.add_column(var, answers, to_metas=True)
        self.Outputs.data.send(data)

    def ask_llm(self,state) -> List:
        if not self._data or not self.text_var:
            return []

        texts = self._data.get_column(self.text_var)
        answers = []
        for i, text in enumerate(texts):
            
            state.set_progress_value(i / len(texts) * 100)
            if state.is_interruption_requested():
                raise Exception
            
            #MODELS = ["sshleifer/distilbart-cnn-12-6"]
            
            args = (MODELS[self.model_index],
                    text.strip(), 
                    self.prompt_start.strip(),
                    self.prompt_end.strip())
            
            if args in self.cache:
                answer = self.cache[args]
            else:
                try:
                    answer = self.run_summarizer_model(f"{self.prompt_start}\n{text.strip()}\n{self.prompt_end}")
                    self.cache[args] = answer
                except Exception as ex:
                    answer = str(ex)
            answers.append(answer)
        return answers
    
if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from Orange.data import Table

    WidgetPreview(OWStorySetting).run(set_data=Table("zoo"))