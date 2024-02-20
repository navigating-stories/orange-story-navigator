from transformers import pipeline

class SentimentAnalyzer:
    """Class to perform sentiment analysis of textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.7/
    """

    def __init__(self):
        self.sentiment_model = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
            top_k=None
        )

    def __compute_individual_sentiment_score(self, sentence):
        result = self.sentiment_model(sentence)[0]
        for i in range(0, 3):
            if result[i]['label'] == 'positive':
                p_score = result[i]['score']
            if result[i]['label'] == 'neutral':
                neu_score = result[i]['score']
            if result[i]['label'] == 'negative':
                neg_score = result[i]['score']

        overall_score = (p_score - neg_score) / (p_score + neg_score + neu_score)
        return {'positive' : p_score, 'neutral' : neu_score, 'negative' : neg_score, 'overall' : overall_score} 

    def compute_sentiment_scores(self, story_elements, callback=None):
        # Load model directly
        # from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
        # model = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")

        story_els = story_elements.copy()
        sentences = story_els['sentence'].unique().tolist()
        self.sentiment_scores_dict_pos = {}
        self.sentiment_scores_dict_neg = {}
        self.sentiment_scores_dict_neu = {}
        self.sentiment_scores_dict_over = {}
        c = 1
        for sentence in sentences:
            current_scores = self.__compute_individual_sentiment_score(sentence)
            self.sentiment_scores_dict_pos[sentence] = current_scores['positive']
            self.sentiment_scores_dict_neu[sentence] = current_scores['neutral']
            self.sentiment_scores_dict_neg[sentence] = current_scores['negative']
            self.sentiment_scores_dict_over[sentence] = current_scores['overall']
            c += 1
            if callback:
                callback((c/len(sentences))*100)

        story_els['positive_sent'] = story_els['sentence'].map(self.sentiment_scores_dict_pos)
        story_els['negative_sent'] = story_els['sentence'].map(self.sentiment_scores_dict_neg)
        story_els['neutral_sent'] = story_els['sentence'].map(self.sentiment_scores_dict_neu)
        story_els['overall_sent'] = story_els['sentence'].map(self.sentiment_scores_dict_over)

        return story_els
