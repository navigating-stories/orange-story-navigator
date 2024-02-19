""" Utility function module 
"""
from bs4 import BeautifulSoup
import re
import spacy
import os
import string
import pandas as pd
import storynavigation.modules.constants as constants

def entity_tag_already_exists(ents, start, end):
    for ent in ents:
        if (ent['start'] == start and ent['end'] == end):
            return True
    return False

def get_column(df, entity):
    """Returns the first column number / title in a dataframe in which the given value / item appears

    Args:
        df (pandas dataframe): the dataframe in which to search for the right column
        entity (string): the entity we are search for in the dataframe


    Returns:
        string: column name / number
    """

    for col in df.columns:
        tmplst = [x.lower() for x in df[col].tolist()]
        for item in tmplst:
            if entity.lower() == item:
                return col
    return '' # no column contains the given entity (hopefully impossible)

def remove_duplicate_tagged_entities(ents):
    entities_minus_duplicates = []
    for entity in ents:
        if not entity_tag_already_exists(entities_minus_duplicates, entity['start'], entity['end']):
            entities_minus_duplicates.append(entity)
    return entities_minus_duplicates

def get_normalized_token(token):
    """cleans punctuation from token and verifies length is more than one character

    Args:
        token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

    Returns:
        string: cleaned token text
    """

    if type(token) == spacy.tokens.token.Token:
        normalised_token = token.text.lower().strip()
    else:
        normalised_token = token[0].lower().strip()
    if len(normalised_token) > 1:
        if normalised_token[len(normalised_token) - 1] in string.punctuation:
            normalised_token = normalised_token[: len(normalised_token) - 1].strip()
        if normalised_token[0] in string.punctuation:
            normalised_token = normalised_token[1:].strip()

    return normalised_token

def load_spacy_pipeline(name):
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
        nlp.add_pipe("sentencizer")
    return nlp


def preprocess_text(text):
    """Preprocesses story text. A lot of stories in the Corona in de stad dataset
    have sentences with no period at the end followed immediately by newline characters.
    This function processes these and other issues to make the resulting text suitable for
    further NLP analysis (e.g. postagging and ner).

    Args:
        text (string): Story text

    Returns:
        list: List of processed string sentences in story text
    """
    # find all regex matches for a newline character immediately followed by uppercase letter
    match_indices = []
    for i in re.finditer("\n[A-Z]", text):
        startindex = i.start()
        match_indices.append(startindex + 1)
    match_indices.append(None)
    # split the text into clauses (based on regex matches) - clauses can be single or multiple sentences
    clauses = [
        text[match_indices[i] : match_indices[i + 1]]
        for i in range(0, len(match_indices) - 1)
    ]
    # clean clauses: remove newlines in the middle of clauses and tokenize them into individual sentences
    cleaned_sentences = []
    for clause in clauses:
        cleaned_clause = clause.replace("\n", " ")
        # tokenize clause into sentences
        sentences = cleaned_clause.split(".")
        for sent in sentences:
            sent_tmp = sent.strip()
            if len(sent_tmp) > 1:
                if sent_tmp[len(sent_tmp) - 1] != ".":
                    sent_tmp += "."  # add a period to end of sentence (if there is not one already)
                cleaned_sentences.append(sent_tmp)

    # Handle the case where input story text is not formatted with newline characters (no newlines at all)
    if len(cleaned_sentences) == 0:
        sents = text.split('.')
        if len(sents) > 0:
            for sent in sents:
                cleaned_sentences.append(sent.strip())
        else:
            return []

    # # remove quotes because it affects the accuracy of POS tagging
    cleaned_sents = []
    for item in cleaned_sentences:
        item = item.replace("`", "").replace("'", "").replace("‘", "").replace("’", "")
        cleaned_sents.append(item)

    return cleaned_sents

def frame_contains_custom_tag_columns(story_elements_df):
    df_filtered = pd.DataFrame()
    if 'token_text_lowercase' in story_elements_df.columns:
        df_filtered = story_elements_df.drop(columns=constants.TAGGING_DATAFRAME_COLUMNNAMES+['token_text_lowercase'])
    else:
        df_filtered = story_elements_df.drop(columns=constants.TAGGING_DATAFRAME_COLUMNNAMES)

    if 'associated_action_lowercase' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['associated_action_lowercase'])

    if len(df_filtered.columns) > 1:
        return True
    return False
    
def remove_custom_tag_columns(df):
    dropcols = []
    for col in df.columns:
        if '-scheme_' in col:
            dropcols.append(col)
    df = df.drop(columns=dropcols)  
    return df
    
def get_custom_tags_list_and_columns(story_elements_df):
    postags = []
    columns = []
    df_filtered = pd.DataFrame()
    if 'token_text_lowercase' in story_elements_df.columns:
        df_filtered = story_elements_df.drop(columns=constants.TAGGING_DATAFRAME_COLUMNNAMES+['token_text_lowercase'])
    else:
        df_filtered = story_elements_df.drop(columns=constants.TAGGING_DATAFRAME_COLUMNNAMES)

    if 'associated_action_lowercase' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['associated_action_lowercase'])
    
    if len(df_filtered.columns) > 1:
        for colname in df_filtered.columns:
            if not colname.startswith('custom_'):
                columns.append(colname)
                postags.extend(df_filtered[colname].unique().tolist())
    postags = list(set(postags))
    return columns, postags

def convert_orangetable_to_dataframe(table):
    """Converts an Orange Data Table object to a Pandas dataframe

    Args:
        table (Orange.data.Table): an Orange Data Table instance

    Returns:
        df (pandas.DataFrame): a pandas dataframe with the same content (info) and structure contained in the Orange Data Table
    """
    # Extract attribute names, class variable name, and meta attribute names
    column_names = [var.name for var in table.domain.variables]
    meta_names = [meta.name for meta in table.domain.metas]

    # Combine attribute and meta names
    all_column_names = column_names + meta_names

    # Create a list of lists representing the data
    data = [[str(entry[var]) for var in table.domain.variables + table.domain.metas] for entry in table]

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data, columns=all_column_names)

    return df

def remove_span_tags(html_string):
    """Removes span tags (including content) from an HTML string

    Args:
        html_string (string) : HTML string

    Returns:
        HTML string without span tags and associated content
    """
    soup = BeautifulSoup(html_string, "html.parser")

    # Remove all <span> tags
    for span_tag in soup.find_all("span"):
        span_tag.decompose()

    return str(soup)

def remove_span_tags_except_custom(html_string):
    """Removes span tags (including content) from an HTML string (except ones for custom tags)

    Args:
        html_string (string) : HTML string

    Returns:
        HTML string without span tags and associated content
    """
    soup = BeautifulSoup(html_string, "html.parser")

    # Remove all <span> tags
    for span_tag in soup.find_all("span"):
        outer_tag = span_tag.find_parent()
        if 'style' in outer_tag.attrs:
            if "background: "+constants.CUSTOMTAG_HIGHLIGHT_COLOR not in outer_tag['style']:
                span_tag.decompose()

    return str(soup)

def is_only_punctuation(phrase):
    if all(char in string.punctuation + str(' ') for char in phrase):
        return '-'
    else:
        return phrase

# Function to recursively traverse ancestors
def find_verb_ancestor(token):
    """Finds the main verb associated with a token (mostly nouns) in a sentence

    Args:
        token (spacy.tokens.token.Token): input token

    Returns:
        verb: the verb text if any, otherwise None
    """
    if isinstance(token, spacy.tokens.token.Token):
        # Check if the token is a verb
        if token.pos_ == "VERB":
            return token

        # Traverse the token's ancestors recursively
        for ancestor in token.ancestors:
            # Recursive call to find the verb ancestor
            verb_ancestor = find_verb_ancestor(ancestor)
            if verb_ancestor:
                return verb_ancestor
    elif isinstance(token, tuple):
        return find_verb_ancestor(token[4])

    # If no verb ancestor found, return None
    return None
