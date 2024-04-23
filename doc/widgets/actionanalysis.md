Actions
=======

![](../../orangecontrib/storynavigation/widgets/icons/action_analysis_icon.png)

The *Action Analysis* widget provides a tool to support basic narrative analysis for actions in stories. It is part of the Orange3-Story-Navigator add-on for the Orange data mining software package. The widget highlights present and past tense actions, calculate their frequency, and identify actors associated with those actions, within in a textual story written in the Dutch language.

**Main Features**
- Analyses actions in stories based on part of speech (POS) tagging and verb tense.
- Allows selection of entity type to highlight (POS or NER).
- Provides options to filter and highlight specific parts of speech.
- Supports copying of analysis results to clipboard.
- Allows for customization of POS checkboxes for each POS type.

**Inputs**

- *Story elements*: The action widget always requires story elements from the elements widget. The widget will not work without this input.
- *Stories*: A dataset of one or more textual story documents in Dutch.
- *Token categories* (**optional**): a data table specifying one or more classification schemes of tokens or words. The table should consist of at least two columns. The first column is a list of words or tokens. All subsequent columns should contain strings which represent user-defined category labels for the corresponding word or token in the first column.

**Outputs**

- *Action stats*: A data table with four columns. 
  - *storyid* is the first column, matching with a particular story from the corpus
  - *segment_id* represents the amount of segments the story has been divided into (see the elements widget)
  - *story navigator tag* describes the verb tense, indicating the time at which an action takes place (i.e., past, present, or future) 
  - *wordcol* describes the frequency of a verb tense per story id.
   
- *Custom tag stats* (**optional**): A data table with five columns.
  - *storyid*, matching with a particular story from the corpus
  - *segment_id* represents the amount of segments the story has been divided into
  - *category* further sub-categorizes all the verbs in a story, with the type of sub-category depending on the  column 'classification'. The specific (sub)-categories can be manually specified, depending on the research interests, and is input for the elements widget. Note that a verb can be part of more than 1 category, depending on the context and quality of the verb.
  - *freq* describes the verb-frequency within each subcategory (i.e., column category), per story id.
  - *classification* is the higher-order category, as manually specified.
    
- *Action table*: A data table with three columns. 
  - *action* specifies all the verbs which occured accross the corpus. Duplicate verbs occur because the action table accounts for different type of entities associated with the action.
  - *entities* specifies all the actors associated with the action from the action column, accross the entire corpus.
  - *entities_type* further specifies the association between action and entity, based on the entity's morphological property (e.g., singular proper noun, noun that is singular and non-proper, etc.).



Example usage:
--------------

![](images/sn_action_analysis_example.png)