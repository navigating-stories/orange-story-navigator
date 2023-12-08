import sys
import os
import pandas as pd
from operator import itemgetter
import spacy
import storynavigation.modules.constants as constants
import re
from spacy.lang.nl import Dutch
from spacy import displacy
import string
from nltk.tokenize import RegexpTokenizer
from thefuzz import fuzz
from thefuzz import process

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources

class ActorTagger:
    """ Class to perform NLP analysis of actors in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.7/
    """

    PKG = importlib_resources.files(constants.MAIN_PACKAGE)
    NL_STOPWORDS_FILE = PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_STOPWORDS_FILENAME
    NL_PRONOUNS_FILE = PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_PRONOUNS_FILENAME

    def __init__(self, model):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        pr = self.NL_PRONOUNS_FILE.read_text(encoding="utf-8")
        self.pronouns = pr
        self.stopwords = s
        self.html_result = ''
        
        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        self.nlp = self.__load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.
        self.agent_prominence_score_min = 0.

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0

    
    @classmethod
    def __load_spacy_pipeline(self, name):
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
    
    def __preprocess_text(self, text):
        # find all regex matches for a newline character immediately followed by uppercase letter
        match_indices = []
        for i in re.finditer(u'\n[A-Z]', text):
            startindex= i.start()
            match_indices.append(startindex+1)
        match_indices.append(None)
        # split the text into clauses (based on regex matches) - clauses can be single or multiple sentences
        clauses = [text[match_indices[i]:match_indices[i+1]] for i in range(0,len(match_indices)-1)]
        # clean clauses: remove newlines in the middle of clauses and tokenize them into individual sentences
        cleaned_sentences = []
        for clause in clauses:
            cleaned_clause = clause.replace('\n', ' ')
            # tokenize clause into sentences
            sentences = cleaned_clause.split('.')
            for sent in sentences:
                sent_tmp = sent.strip()
                if len(sent_tmp) > 1:
                    if sent_tmp[len(sent_tmp)-1] != '.':
                        sent_tmp += '.' # add a period to end of sentence (if there is not one already)
                    cleaned_sentences.append(sent_tmp)

        return cleaned_sentences
    
    def nertag_text(self, text, per, loc, product, date, nlp):
        sentences = self.__preprocess_text(text)
        html = ""

        ner_tags = []
        if (per):
            ner_tags.append("PERSON")
        if (loc):
            ner_tags.append("LOC")
            ner_tags.append("GPE")
            ner_tags.append("NORP")
            ner_tags.append("FAC")
            ner_tags.append("ORG")
            ner_tags.append("EVENT")
        if (product):
            ner_tags.append("PRODUCT")
            ner_tags.append("WORK_OF_ART")
        if (date):
            ner_tags.append("DATE")
            ner_tags.append("TIME")

        options = {"ents": ner_tags, "colors": {}}

        for sent in sentences:
            tagged_sentence = nlp(sent)
            html += displacy.render(tagged_sentence,
                                    style="ent", options=options)

        return html

    def __update_postagging_metrics(self, tagtext, tags, selected_prominence_metric, agent_prominence_score_min, token):
        vb = self.__find_verb_ancestor(token)
        if vb is not None:
            if tagtext in self.noun_action_dict:
                self.noun_action_dict[tagtext].append(vb.text)
            else:
                self.noun_action_dict[tagtext] = []

        p_score = self.__calculate_prominence_score(tagtext, selected_prominence_metric)
        self.word_prominence_scores[tagtext] = p_score

        if (p_score >= agent_prominence_score_min):
            return True
        else:
            return False
    
    def __calculate_pretagging_metrics(self, sentences, stopwords=None):
        self.sentence_count = len(sentences)
        for sentence in sentences:
            words = sentence.split(' ')
            tokens = []
            for word in words:
                if len(word) > 1:
                    if word[len(word)-1] in string.punctuation:
                        tokens.append(word[:len(word)-1].lower().strip())
                    else:
                        tokens.append(word.lower().strip())

            self.word_count += len(tokens)

            if stopwords is not None and len(stopwords) > 0:
                for token in tokens:
                    if token not in stopwords:
                        self.word_count_nostops += 1
            else:
                self.word_count_nostops = self.word_count

    def __is_subject(self, tag):
        if (tag[3].lower() in ['nsubj', 'nsubj:pass', 'csubj'] and tag[1] in ['PRON', 'NOUN', 'PROPN']):
            if tag[1] == 'PRON':
                return True, 'PRON'
            elif tag[1] == 'NOUN':
                return True, 'NOUN'
            else:
                return True, 'PROPN'
        return False, ''
    
    def __is_pronoun(self, tag, stopwords):
        if (tag[0].lower().strip() == 'ik'):
            return True
        if tag[0].lower().strip() not in stopwords:
            if tag[1] == 'PRON':
                if ('|' in tag[2]):
                    tmp_tags = tag[2].split('|')
                    if (tmp_tags[1] == 'pers' and tmp_tags[2] == 'pron') or (tag[0].lower().strip() == 'ik'):
                        return True
        return False
    
    def __is_noun_but_not_pronoun(self, tag, stopwords):
        if ((not self.__is_pronoun(tag, stopwords)) and (tag[1] in ['NOUN', 'PROPN'])):
            return True
        else:
            return False
    
    def postag_text(self, text, vbz, adj, nouns, subjs, nlp, stopwords, selected_prominence_metric, agent_prominence_score_min):
        sentences = self.__preprocess_text(text)
        self.__calculate_pretagging_metrics(sentences, stopwords=stopwords)

        # pos tags that the user wants to highlight
        pos_tags = []

        # add pos tags to highlight according to whether the user has selected them or not
        if (vbz):
            pos_tags.append("VERB")
        if (adj):
            pos_tags.append("ADJ")
            pos_tags.append("ADV")
        if (nouns):
            pos_tags.append("NOUN")
            pos_tags.append("PRON")
            pos_tags.append("PROPN")
        if (subjs):
            pos_tags.append("SUBJ")

        # output of this function
        html = ""

        # generate and store nlp tagged models for each sentence
        sentence_nlp_models = []
        for sentence in sentences:
            tagged_sentence = nlp(sentence)
            sentence_nlp_models.append(tagged_sentence)
 
        self.__calculate_word_type_count(sentences, sentence_nlp_models, stopwords=stopwords)

        # loop through model to filter out those words that need to be tagged (based on user selection and prominence score)
        for sentence, tagged_sentence in zip(sentences, sentence_nlp_models):
            first_word_in_sent = sentence.split()[0].lower().strip()
            tags = []
            tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
            spans = list(tokenizer.span_tokenize(sentence))

            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_, token.dep_))

            ents = []
            for tag, span in zip(tags, spans):
                normalised_token, is_valid_token = self.__is_valid_token(tag, stopwords)
                if is_valid_token:
                    is_subj, subj_type = self.__is_subject(tag)
                    if is_subj:
                        print()
                        print('sentence: ', sentence)
                        print()
                        print('word: ', tag[0].lower().strip())
                        print()
                        p_score_greater_than_min = self.__update_postagging_metrics(tag[0].lower().strip(), tags, selected_prominence_metric, agent_prominence_score_min, token)
                        if (p_score_greater_than_min):
                            ents.append({"start": span[0], "end": span[1],"label": "SUBJ"})
                    else:
                        if self.__is_pronoun(tag, stopwords) or self.__is_noun_but_not_pronoun(tag, stopwords):
                            ents.append({"start": span[0],"end": span[1],"label": "NOUN"})

            if first_word_in_sent in self.pronouns:
                p_score_greater_than_min = self.__update_postagging_metrics(first_word_in_sent, tags, selected_prominence_metric, agent_prominence_score_min, token)
                if (p_score_greater_than_min):
                    ents.append({"start": 0,"end": len(first_word_in_sent),"label": "SUBJ"})            
                
            # specify sentences and filtered entities to tag / highlight
            doc = {"text": sentence, "ents": ents}

            # specify colors for highlighting each entity type
            colors = {}
            if nouns:
                colors["NOUN"] = constants.NOUN_HIGHLIGHT_COLOR
            if subjs:
                colors["SUBJ"] = constants.SUBJECT_HIGHLIGHT_COLOR

            self.agent_prominence_score_max = self.__get_max_prominence_score()
            # collect the above config params together
            options = {"ents": pos_tags, "colors": colors}
            # give all the params to displacy to generate HTML code of the text with highlighted tags
            html += displacy.render(doc, style="ent",options=options, manual=True)

        self.html_result = html

        return html
    
    def __get_normalized_token(self, token):
        if type(token) == spacy.tokens.token.Token:
            normalised_token = token.text.lower().strip()
        else:
            normalised_token = token[0].lower().strip()
        if len(normalised_token) > 1:
            if normalised_token[len(normalised_token)-1] in string.punctuation:
                normalised_token = normalised_token[:len(normalised_token)-1]
        return normalised_token

    def __is_valid_token(self, token, stopwords):
     word = self.__get_normalized_token(token)
     return word, (word not in stopwords) and len(word) > 1

    def __calculate_word_type_count(self, sents, sent_models, stopwords):
        for sent_model in sent_models:
            for token in sent_model:
                normalised_token, is_valid_token = self.__is_valid_token(token, stopwords)
                tag = (token.text, token.pos_, token.tag_, token.dep_)
                if is_valid_token:
                    is_subj, subj_type = self.__is_subject(tag)
                    if is_subj:
                        if token.text.lower().strip() in self.num_occurences_as_subject:
                            self.num_occurences_as_subject[token.text.lower().strip()] += 1
                        else:
                            self.num_occurences_as_subject[token.text.lower().strip()] = 1                        
                    else:
                        if self.__is_pronoun(tag, stopwords) or self.__is_noun_but_not_pronoun(tag, stopwords):
                            if token.text.lower().strip() in self.num_occurences:
                                self.num_occurences[token.text.lower().strip()] += 1
                            else:
                                self.num_occurences[token.text.lower().strip()] = 1

        for sent in sents:
            word = sent.split()[0].lower().strip()
            if word in self.pronouns:
                if word in self.num_occurences_as_subject:
                    self.num_occurences_as_subject[word] += 1
                else:
                    self.num_occurences_as_subject[word] = 1

    def __find_closest_match(self, word, dictionary):
        highest_score = -10
        word_with_highest_score = word
        for item in dictionary:
            similarity_score = fuzz.ratio(item, word)
            if similarity_score > highest_score:
                highest_score = similarity_score
                word_with_highest_score = item

        if highest_score > 80:
            return word_with_highest_score, True
        else:
            return word, False


    def __calculate_prominence_score(self, word, selected_prominence_metric):
        score = 0
        # match spacy-tagged token text to the existing dictionary of words in num_occurrences_as_subject
        closest_match_word, successful_match = self.__find_closest_match(word, self.num_occurences_as_subject)
        if (selected_prominence_metric == "Subject frequency (normalized)"):
            score = (0.7 * (self.num_occurences_as_subject[closest_match_word] / self.word_count)) + (0.3 * (self.num_occurences[closest_match_word] / self.word_count))
        elif (selected_prominence_metric == "Subject frequency"):
            score = self.num_occurences_as_subject[closest_match_word]
        else:
            score = 0
    
        return score
    
    # Function to recursively traverse ancestors
    def __find_verb_ancestor(self, token):
        # Check if the token is a verb
        if token.pos_ == 'VERB':
            return token

        # Traverse the token's ancestors recursively
        for ancestor in token.ancestors:
            # Recursive call to find the verb ancestor
            verb_ancestor = self.__find_verb_ancestor(ancestor)
            if verb_ancestor:
                return verb_ancestor

        # If no verb ancestor found, return None
        return None

    def __get_max_prominence_score(self):
        highest_score = 0
        for item in self.word_prominence_scores:
            if self.word_prominence_scores[item] > highest_score:
                highest_score = self.word_prominence_scores[item]
        return highest_score

    def calculate_agency_table(self):
        rows = []
        n = 10
        res = dict(sorted(self.num_occurences_as_subject.items(),
                   key=itemgetter(1), reverse=True)[:n])
        words = list(res.keys())
        subj_count_values = list(res.values())

        for word in words:
            match, foundmatch = self.__find_closest_match(word, self.num_occurences)
            if word in self.num_occurences:
                rows.append([word, self.num_occurences_as_subject[word], self.num_occurences[word]])
            elif foundmatch:
                rows.append([word, self.num_occurences_as_subject[word], self.num_occurences[match]])
            else:
                rows.append([word, self.num_occurences_as_subject[word], 0])

        return pd.DataFrame(rows, columns=constants.AGENCY_TABLE_HEADER)

    def generate_noun_action_table(self):
        n = 10
        res = dict(sorted(self.word_prominence_scores.items(),
                   key=itemgetter(1), reverse=True)[:n])
        
        names = list(res.keys())

        rows = []
        for item in self.noun_action_dict:
            if len(self.noun_action_dict[item]) > 0 and (item in names):
                curr_row = []
                curr_row.append(item)
                curr_row.append(', '.join(list(set(self.noun_action_dict[item]))))
                rows.append(curr_row)

        return pd.DataFrame(rows, columns=['actor', 'actions'])
    
    @staticmethod
    def sort_tuple(tup):
        lst = len(tup)
        for i in range(0, lst):
            for j in range(0, lst-i-1):
                if (tup[j][1] > tup[j + 1][1]):
                    temp = tup[j]
                    tup[j] = tup[j + 1]
                    tup[j + 1] = temp
        return tup
    

class ActorMetricCalculator:
    def __init__(self, text, listofwords):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        self.stopwords = s
        self.html_result = ''
        
        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        self.nlp = self.__load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.
        self.agent_prominence_score_min = 0.

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0
