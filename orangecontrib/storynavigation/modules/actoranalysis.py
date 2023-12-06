import sys
import os
import pandas as pd
from operator import itemgetter
import spacy
import storynavigation.modules.constants as constants
import re
from spacy.lang.nl import Dutch
from spacy import displacy

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

    def __init__(self, model):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        self.stopwords = s
        self.html_result = ''
        
        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.count_per_word = {}
        self.count_per_subject = {}
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
    
    def nertag_text(self, text, per, loc, product, date, nlp):
        doc = nlp(text)
        sents = list(doc.sents)
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
        if (product):
            ner_tags.append("ORG")
            ner_tags.append("PRODUCT")
            ner_tags.append("EVENT")
            ner_tags.append("WORK_OF_ART")
        if (date):
            ner_tags.append("DATE")
            ner_tags.append("TIME")

        options = {"ents": ner_tags, "colors": {}}

        for span in sents:
            sentence = span.text.replace("\n", " ")
            sentence = sentence.replace("  ", " ")
            sentence = re.sub('\s+', ' ', sentence)
            sentence = sentence.replace('.', '')
            tagged_sentence = nlp(sentence)
            html += displacy.render(tagged_sentence,
                                    style="ent", options=options)

        return html

    
    def postag_text(self, text, vbz, adj, nouns, subjs, nlp, stopwords, selected_prominence_metric, agent_prominence_score_min):
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

        # tokenize input text into sentences
        nlp_nl = Dutch()
        tokenizer = nlp_nl.tokenizer
        nlp_nl.add_pipe("sentencizer")
        doc = nlp_nl(text)
        sents_spans = list(doc.sents)
        sents = []
        for span in sents_spans:
            # sentence = span.text.replace("\n", " ")
            sentence = span.text.replace("  ", " ")
            # sentence = re.sub('\s+', ' ', sentence)
            # sentence = sentence.replace('.', '')
            sents.append(sentence)

        # count no. of sents
        sentence_count = len(sents)
        # print()
        # print(sentence_count)
        # print()
        # count no. of words and words not considering stopwords
        for sentence in sents:
            tokens_doc = tokenizer(sentence)
            tokens = []
            for d in tokens_doc:
                tokens.append(d.text)

            self.word_count += len(tokens)
            for token in tokens:
                if token.lower().strip() in self.sentence_count_per_word:
                    self.sentence_count_per_word[token.lower().strip()] += 1
                else:
                    self.sentence_count_per_word[token.lower().strip()] = 0
                if token.lower().strip() in self.count_per_word:
                    self.count_per_word[token.lower().strip()] += 1
                else: 
                    self.count_per_word[token.lower().strip()] = 0

                if token.lower().strip() in self.count_per_subject:
                    self.count_per_subject[token.lower().strip()] += 1
                else: 
                    self.count_per_subject[token.lower().strip()] = 0

                self.noun_action_dict[token.lower().strip()] = []

                if token.lower().strip() not in stopwords:
                    self.word_count_nostops += 1

        # output of this function
        html = ""

        # generate and store nlp tagged models for each sentence
        sentence_nlp_models = []
        for sentence in sents:
            tagged_sentence = nlp(sentence)
            sentence_nlp_models.append(tagged_sentence)
            # for token in tagged_sentence:
                # self.count_per_word[token.text.lower().strip()] = 0
                # self.count_per_subject[token.text.lower().strip()] = 0
 

        # calculate the number of unique nouns in the text
        self.__calculate_word_type_count(sentence_nlp_models, stopwords=stopwords)

        # loop through model to filter out those words that need to be tagged (based on user selection and prominence score)
        for sentence, tagged_sentence in zip(sents, sentence_nlp_models):
            tags = []
            subjs = []
            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_, token.dep_))
                if (token.dep_ == 'nsubj' and token.pos_ in ['PRON', 'NOUN', 'PROPN']) or token.text.lower().strip() == 'ik':
                    subjs.append((token, token.idx, token.idx+len(token)))

            subjs = self.sort_tuple(subjs)
            main_subject = ''
            
            if len(subjs) > 0:
                main_subject = subjs[0][0].text

            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
            spans = list(tokenizer.span_tokenize(sentence))

            ents = []
            for tag, span in zip(tags, spans):

                if tag[0].lower().strip() not in stopwords:
                    if tag[1] == 'PRON':
                        if ('|' in tag[2]):
                            tmp_tags = tag[2].split('|')
                            if (tmp_tags[1] == 'pers' and tmp_tags[2] == 'pron') or (tag[0].lower().strip() == 'ik'):
                                p_score = 0
                                p_score = self.__calculate_prominence_score(
                                    tag[0].lower().strip(), sents, tags, selected_prominence_metric)
                                self.word_prominence_scores[tag[0].lower(
                                ).strip()] = p_score

                                if (p_score >= agent_prominence_score_min):
                                    if tag[0].lower().strip() == main_subject.lower().strip():
                                        ents.append({"start": span[0],
                                                     "end": span[1],
                                                     "label": "SUBJ"})
                                    else:
                                        ents.append({"start": span[0],
                                                     "end": span[1],
                                                     "label": tag[1]})

                                vb = self.__find_verb_ancestor(token)
                                if vb is not None:
                                    self.noun_action_dict[tag[0].lower().strip()].append(
                                        vb.text)

                    elif ((tag[1] == 'NOUN') or (tag[1] == 'PROPN')):
                        p_score = 0
                        p_score = self.__calculate_prominence_score(tag[0].lower().strip(), sents, tags, selected_prominence_metric)
                        self.word_prominence_scores[tag[0].lower(
                        ).strip()] = p_score

                        if (p_score >= agent_prominence_score_min):
                            if tag[0].lower().strip() == main_subject.lower().strip():
                                ents.append({"start": span[0],
                                             "end": span[1],
                                             "label": "SUBJ"})
                            else:
                                ents.append({"start": span[0],
                                             "end": span[1],
                                             "label": tag[1]})

                        vb = self.__find_verb_ancestor(token)
                        if vb is not None:
                            self.noun_action_dict[tag[0].lower().strip()].append(
                                vb.text)

                    else:
                        ents.append({"start": span[0],
                                    "end": span[1],
                                     "label": tag[1]})

            # specify sentences and filtered entities to tag / highlight
            doc = {"text": sentence, "ents": ents}

            # specify colors for highlighting each entity type
            colors = {}
            if nouns:
                colors["NOUN"] = "turquoise"
                colors["PRON"] = "#BB4CBA"
                colors["PROPN"] = "#259100"
            if subjs:
                colors["SUBJ"] = "#FFEB26"
            if vbz:
                colors["VERB"] = "lightpink"
            if adj:
                colors["ADJ"] = "lime"
                colors["ADP"] = "khaki"
                colors["ADV"] = "orange"

            self.agent_prominence_score_max = self.__get_max_prominence_score()
            # collect the above config params together
            options = {"ents": pos_tags, "colors": colors}
            # give all the params to displacy to generate HTML code of the text with highlighted tags
            html += displacy.render(doc, style="ent",
                                    options=options, manual=True)

        self.html_result = html

        return html

    def __calculate_word_type_count(self, sent_models, stopwords):
        for sent_model in sent_models:
            tags = []
            subjs = []
            
            for token in sent_model:
                if (token.text.lower().strip() not in stopwords):
                    if (token.dep_ == 'nsubj' and token.pos_ in ['PRON', 'NOUN', 'PROPN']) or token.text.lower().strip() == 'ik':
                        subjs.append((token, token.idx, token.idx+len(token)))
                    else:
                        self.count_per_word[token.text.lower().strip()] += 1

            subjs = self.sort_tuple(subjs)
            main_subject = ''
            
            if len(subjs) > 0:
                main_subject = subjs[0][0].text
                self.count_per_subject[main_subject.lower().strip()] += 1
                self.count_per_word[main_subject.lower().strip()] += 1

    def __calculate_prominence_score(self, word, list_of_sentences, tags, selected_prominence_metric):
        score = 0
        word_count = 0
        for sent in list_of_sentences:
            items = sent.split()
            word_count+= len(items)

        if (selected_prominence_metric == "Subject frequency (normalized)"):
            score = (1 - ((self.count_per_word[word] - self.count_per_subject[word]) / self.count_per_word[word])) * (self.count_per_word[word] / word_count) * 100
        elif (selected_prominence_metric == "Subject frequency"):
            score = self.count_per_subject[word]
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
        res = dict(sorted(self.count_per_subject.items(),
                   key=itemgetter(1), reverse=True)[:n])
        words = list(res.keys())
        subj_count_values = list(res.values())

        for word in words:
            rows.append([word, self.count_per_subject[word], self.count_per_word[word]])
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
        self.count_per_word = {}
        self.count_per_subject = {}
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
