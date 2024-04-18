"Define error messages to be used in the Actor and Action analysis"

from Orange.widgets.widget import Msg

msg_wrong_input_for_stories = (
    "Wrong input to `Stories`. "
    "The input to `Story elements` needs to be a `Table`. \n"
    "The input to `Stories` needs to be a `Corpus`." 
)

msg_wrong_input_for_elements = (
    "Wrong input to `Story elements`. "
    "The input to `Story elements` needs to be a `Table`. \n"
    "The input to `Stories` needs to be a `Corpus`."
) 

msg_wrong_story_input_for_elements = (
    "Wrong input to `Stories`. "
    "The input to `Stories` needs to be a `Corpus`. \n"
    "The input to `Custom tags` needs to be a `Table`."
) 

msg_residual_error = (
    "Could not process data. Check the inputs to the widget."
)

wrong_input_for_stories = Msg(msg_wrong_input_for_stories)
residual_error = Msg(msg_residual_error)
wrong_input_for_elements = Msg(msg_wrong_input_for_elements)
wrong_story_input_for_elements = Msg(msg_wrong_story_input_for_elements)


__all__ = [
    "wrong_input_for_stories", "wrong_input_for_elements", "residual_error"
]
