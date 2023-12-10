# metrics for measuring importance of characters or actors in the story
AGENT_PROMINENCE_METRICS = ['Subject frequency', 'Subject frequency (normalized)']
SFREQ_METRIC = 'Subject frequency'
SFREQ_NORM_METRIC = 'Subject frequency (normalized)'
# list of punctuation characters
PUNC = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
# name of storynavigator package directory
MAIN_PACKAGE = "storynavigation"
# directory name for resource files for usage by storynavigator add-on
RESOURCES_SUBPACKAGE = "resources"
# filename from which to retrieve a list of dutch stopwords
NL_STOPWORDS_FILENAME = "dutchstopwords.txt"
# filename from which to retrieve a list of dutch stopwords
NL_PRONOUNS_FILENAME = "dutchpronouns.txt"
# currently selected agent prominence metric
SELECTED_PROMINENCE_METRIC = 'Subject frequency'
# column names for agency table
FREQ_TABLE_HEADER = ['actor', 'raw_frequency']
SUBFREQ_TABLE_HEADER = ['actor', 'subject_frequency']
AGENCY_TABLE_HEADER = ['actor', 'agency']
# Halliday dimensions file
HALLIDAY_FILENAME = "halliday_dimensions_{}.json"
# dutch spacy model (small)
NL_SPACY_MODEL = "nl_core_news_sm"

# colors for highlighting words in text
# NOUN_HIGHLIGHT_COLOR = "turquoise"
# PRONOUN_HIGHLIGHT_COLOR = "#BB4CBA"
# PROPERNOUN_HIGHLIGHT_COLOR = "#259100"
SUBJECT_PRONOUN_HIGHLIGHT_COLOR = "#87CEFA"
SUBJECT_NONPRONOUN_HIGHLIGHT_COLOR = "#ADD8E6"
NONSUBJECT_PRONOUN_HIGHLIGHT_COLOR = "#FFA500"
NONSUBJECT_NONPRONOUN_HIGHLIGHT_COLOR = "#FFE4B5"
	#ADD8E6
# VERB_HIGHLIGHT_COLOR = "lightpink"
# ADJECTIVE_HIGHLIGHT_COLOR = "lime"
# ADVERB_HIGHLIGHT_COLOR = "khaki"
# ADPOSITION_HIGHLIGHT_COLOR = "orange"