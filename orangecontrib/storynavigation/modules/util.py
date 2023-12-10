""" Utility function module 
"""
from bs4 import BeautifulSoup
import re
import spacy
import os
import string


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

    return cleaned_sentences


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


# Function to recursively traverse ancestors
def find_verb_ancestor(token):
    """Finds the main verb associated with a token (mostly nouns) in a sentence

    Args:
        token (spacy.tokens.token.Token): input token

    Returns:
        verb: the verb text if any, otherwise None
    """
    # Check if the token is a verb
    if token.pos_ == "VERB":
        return token

    # Traverse the token's ancestors recursively
    for ancestor in token.ancestors:
        # Recursive call to find the verb ancestor
        verb_ancestor = find_verb_ancestor(ancestor)
        if verb_ancestor:
            return verb_ancestor

    # If no verb ancestor found, return None
    return None
