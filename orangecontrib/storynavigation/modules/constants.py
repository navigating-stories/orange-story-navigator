# metrics for measuring importance of characters or actors in the story
AGENT_PROMINENCE_METRICS = ['Subject frequency', 'Subject frequency (normalized)']
# list of punctuation characters
PUNC = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
# name of storynavigator package directory
MAIN_PACKAGE = "storynavigation"
# directory name for resource files for usage by storynavigator add-on
RESOURCES_SUBPACKAGE = "resources"
# filename from which to retrieve a list of dutch stopwords
NL_STOPWORDS_FILENAME = "dutchstopwords.txt"
# currently selected agent prominence metric
SELECTED_PROMINENCE_METRIC = 'Subject frequency'
# column names for agency table
AGENCY_TABLE_HEADER = ['actor', 'subject_frequency', 'raw_frequency']
# Halliday dimensions file
HALLIDAY_FILENAME = "halliday_dimensions_{}.json"
# dutch spacy model (small)
NL_SPACY_MODEL = "nl_core_news_sm"