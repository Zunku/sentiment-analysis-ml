
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from utilities.text_utils import re_breakline, re_dates, re_hiperlinks, re_money, re_negation, re_numbers, \
    re_special_chars, re_whitespaces, ApplyRegex, StemmingProcess, StopWordsRemoval
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Function that apply Regular Expressions, stopwords and other techniques to comments reviews
def text_transformers():
    # Defining regex transformers to be applied
    regex_transformers = {
        'break_line': re_breakline,
        'hiperlinks': re_hiperlinks,
        'dates': re_dates,
        'money': re_money,
        'numbers': re_numbers,
        'negation': re_negation,
        'special_chars': re_special_chars,
        'whitespaces': re_whitespaces
    }

    # Building a text prep pipeline
    text_prep_pipeline = Pipeline([
        ('regex', ApplyRegex(regex_transformers)),
        ('stopwords', StopWordsRemoval(stopwords.words('portuguese'))),
        ('stemming', StemmingProcess(RSLPStemmer())),
        ('vectorizer', TfidfVectorizer(max_features=300, min_df=7, max_df=0.8, stop_words=stopwords.words('portuguese')))
    ])

    #X_prep = text_prep_pipeline.fit_transform(text_list)
    return text_prep_pipeline
