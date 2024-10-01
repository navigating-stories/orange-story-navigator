Actors
======

![](../../orangecontrib/storynavigation/widgets/icons/actor_analysis_icon.png)

An Orange3 widget to highlight the main subjects of sentences as well as other potential actors or characters in a textual story written in the Dutch language.

**Main Features**
- POS Tagging: Perform part-of-speech (POS) tagging on the text of stories, tagging noun tokens and subject tokens as per user specifications.
- Custom Tagging: Highlight custom tags specified by the user within the story text.
- HTML Output: Returns HTML string representations of the POS tagged text, ready for rendering in the UI.
- Frequency Calculation: Prepares data tables for frequencies of custom tokens specified by the user.
- Actor Analysis Results: Generates actor analysis results including raw frequency, subject frequency, agency, and prominence score for each identified actor.

**Inputs**

- *Stories*: A corpus or dataset of one or more textual story documents in Dutch.
  
- *Story elements*: The action widget requires story elements from the elements widget. These story elements refer to the attributes extracted from the textual stories, including linguistic features, such as words, sentences, parts of speech, and syntactic structures, along with additional metadata to contextualize the analysis. These elements enable the actor widget to perform detailed natural language processing and semantic analysis on the textual stories, facilitating the identification and characterization of actors, actions, and relationships within the narratives.

**Outputs**

- *Custom tag stats* (**optional**): A data table with five columns.
  - *storyid*, matching with a particular story from the corpus
  - *segment_id* represents the amount of segments the story has been divided into
  - *category* further sub-categorizes all the verbs in a story, with the type of sub-category depending on the  column 'classification'. The specific (sub)-categories can be manually specified, depending on the research interests, and is input for the elements widget. Note that a verb can be part of more than 1 category, depending on the context and quality of the verb.
  - *freq* describes the verb-frequency within each subcategory (i.e., column category), per story id.
  - *classification* is the higher-order category, as manually specified and input for the elements widget.  

- *Actor stats*: The elements widget generates a data table containing tagging data for all stories processed. Each row in the datatable represents a tagged token within a sentence, providing comprehensive information for further analysis and interpretation. It includes the following columns:

  - *token_text_lowercase*: The text of the token.
  - *storyid*: Unique identifier for the story.
  - *segment_id*: Identifier for the story segment.
  - *raw_freq*: Raw frequency of the custom token in the story segment.
  - *subj_freq*: Subject frequency of the custom token in the story segment.
  - *agency*: of the custom token in the story segment represents the extent of an entity's involvement or influence within the narrative. [Agency](https://journals.sagepub.com/doi/full/10.1177/0081175012462370?casa_token=Lx4o-GJ8wbAAAAAA%3AbolGvtXBrf_Wa84jvVSd02kCt4rXwCGs108iqHk0LoXo1nRMPKnsZwhumUtArpnk_hvJzNiyO7nL5w) measures how actively a particular entity (such as a character, organization, or concept) is engaged in actions or events described in the story. Higher agency values indicate that the entity is more actively involved in driving the narrative forward.
    - The agency is calculated as the ratio of the total occurrences of the entity being the subject of sentences to the total number of sentences in which the entity appears.
  - *prominence_sf*: is the prominence score of the custom token in the story segment, measuring the entity's significance relative to others in the story. Higher prominence scores suggest that the entity plays a more crucial or central role in the narrative.
    - The prominence score is calculated based on the relative frequency of the entity's appearance in subject positions across sentences in the story. It considers both the frequency of occurrence and the distribution of the entity's mentions throughout the narrative. The prominence score calculation involves normalization to account for variations in story length and ensures comparability across different stories.
  
Example usage:
--------------

![](images/sn_actor_analysis_example.png)