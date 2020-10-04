from func import create_csv

import string
import re
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 100)
display_settings = {
    'expand_frame_repr': True,  # Развернуть на несколько страниц
    'precision': 2,
    'show_dimensions': True
}
for op, value in display_settings.items():
    pd.set_option("display.{}".format(op), value)
from collections import Counter
from textatistic import Textatistic
from langdetect import detect_langs
from selenium import webdriver
# For using selenium chromedriver with Google Colab uncomment this line
# sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
# Load the small English model – spaCy is already imported
# nlp = spacy.load('en_core_web_sm')
import en_core_web_md
nlp = en_core_web_md.load()

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = stopwords.words('english')
# Import NLTK Lemmatizer and Stemmer
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

from stop_words import get_stop_words
stop_words_en = get_stop_words('en')
stop_words_ru = get_stop_words('russian')
# For using ukrainian stopwords with Google Colab uncomment this line
# stopwords_ua = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/stopwords_ua.txt", header=None, names=['stopwords'])
stopwords_ua = pd.read_csv("stopwords_ua.txt", header=None, names=['stopwords'])
stop_words_ua = list(stopwords_ua.stopwords)

# Preprocessing
data = pd.read_csv('result_csv/data_04-10-2020_12-12.csv', dtype=str)

# Replace text with regexp
cleaned_text_list = []

def clean_with_regex(text):
    for article in text:
        try:
            clean_endlines = re.sub("\.\n", '.+++', article)
            clean_endlines = re.sub("!\n", '!+++', clean_endlines)
            clean_endlines = re.sub(":\n", '+++', clean_endlines)
            clean_endlines = re.sub("\n", ' ', clean_endlines)
            enter_endlines = re.sub("\+{3}", "\n", clean_endlines)
            # Replace more then two links one after the other
            pattern = "[http]\S+\s[http]\S+\s[http]\S+"
            clean_two_http_links = re.sub(pattern, '', enter_endlines)
            # Clean dates in headers
            pattern = "(\d{1,2}\.\d{2}\.\d{4})(.)+(\d{1,2}\/\d{1,2})"
            clean_http_and_pagenumbers = re.sub(pattern, '', clean_two_http_links)
            cleaned_text_list.append(clean_http_and_pagenumbers)
        except:
            cleaned_text_list.append(np.nan)
    return cleaned_text_list


data["cleaned_body_text"] = clean_with_regex(data['txt'])

# create sentences of texts and count them
text_sentences = []
text_sentences_len = []

def get_sentences(cleaned_text):
    for full_text in cleaned_text:
        full_text = [' '.join(nltk.word_tokenize(sentence)) for sentence in
                     str(full_text).replace('?', '.').replace('!', '.'). \
                         replace(':', '.').replace(';', '.').split('.') if len(sentence) > 20]
        text_sentences.append(full_text)
        text_sentences_len.append(len(full_text))
    return text_sentences, text_sentences_len


data['text_sentences'], data['text_sentences_len'] = get_sentences(data.cleaned_body_text)

# Detect languages of texts
languages = []

def get_languages(cleaned_text):
    # Loop over the rows of the dataset and append
    for row in cleaned_text:
        try:
            if row is not None:
                languages.append(detect_langs(row))
        except:
            languages.append(np.nan)
    # Clean the list by splitting
    language = [str(lang).split(':')[0][1:] for lang in languages]
    return language


# # Assign the list to a new feature
data['language'] = get_languages(data.cleaned_body_text)
# print(data)

# NLTK preprocessing, cleaning text, stemming, lemmatizing

# create data frames for languages
en_col = data[data['language'] == 'en']
ru_col = data[data['language'] == 'ru']
ua_col = data[data['language'] == 'uk']

# clean text (remove stopwords, stemming, lemmatizing)
def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    text = [word for word in text if word not in stop_words_ru]
    text = [word for word in text if word not in stop_words_ua]
    return text

data['body_text_nostop'] = data["cleaned_body_text"].apply(lambda x: clean_text(str(x).lower()))


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

data['body_text_stemmed'] = data['body_text_nostop'].apply(lambda x: stemming(x))


def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))

# Most common words
most_common_list_counts = []
most_common_list = []

def get_most_common_words(text_lemmatized):
    for data_row in text_lemmatized:
        # Create the bag-of-words: bow
        stopingwords = {'experience', 'knowledge', 'skill', 'understanding', 'ability', 'able', 'customer', 'work', 'software', 'company', 'make', 'want', 'build', 'business', 'offer', 'document', 'environment', 'technology', 'solution', 'year', 'development', 'python'}
        bow = Counter(data_row)
        word_bow = [(word, cnt) for word, cnt in bow.most_common(20) if word not in stopingwords]
        word_bow_l = [word[0] for word in word_bow]
        if word_bow:
            most_common_list_counts.append(word_bow)
            most_common_list.append(word_bow_l)

        else:
            most_common_list_counts.append(np.nan)
            most_common_list.append(np.nan)
    return most_common_list_counts, most_common_list


data['most_common_words_counts'], data['most_common_words_list'] = get_most_common_words(data['body_text_lemmatized'])

# if data['body_text_lemmatized'][0][0] == '':
#     data['body_text_lemmatized'][0][0] = np.nan
# Count 5 most common words in different columns

mcw_list = list(data['most_common_words_counts'])


# print(data[['mcw_1', 'mcw_1_count', 'mcw_2', 'mcw_2_count', 'mcw_3', 'mcw_3_count', 'mcw_4', 'mcw_4_count', 'mcw_5',
#       'mcw_5_count']])


# Remove long words
def length(column):
    text = [item for item in column if len(item) < 14]
    # [item for row in column for item in row if len(item) > 14]
    return text

data['body_textlemm_nolongwords'] = data['body_text_lemmatized'].apply(lambda x: length(x))

# Create feature text, punctuation, nonstopwords:
#  text message length
#  % of text that is punctuation
#  non stop words

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    try:
        return round(count / (len(text) - text.count(" ")), 3) * 100
    except:
        pass

# Function that returns number of words in a string
def count_words(string):
    # Split the string into words
    words = string.split()

    # Return the number of words
    return len(words)

def tokens_length(cleaned_body_text):
    # Tokenize each item in the review column
    word_tokens = [word_tokenize(review) for review in data.cleaned_body_text]
    # Create an empty list to store the length of reviews
    len_tokens = []
    # Iterate over the word_tokens list and determine the length of each item
    for i in range(len(word_tokens)):
        len_tokens.append(len(word_tokens[i]))
    return len_tokens

# Count proper nouns and nouns

# Returns number of proper nouns
def proper_nouns(text, model=nlp):
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of proper nouns
    return pos.count('PROPN')

# Returns number of other nouns
def nouns(text, model=nlp):
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of other nouns
    return pos.count('NOUN')

def find_persons(text):
    # Create Doc object
    doc = nlp(text)

    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Return persons
    return set(persons)


# Create a feature for count body and cleaned body length, words, punctuation, characters, nostopwords, stemmed and lemmatized words
data['body_len'] = data["txt"].apply(lambda x: len(str(x)) - str(x).count(" "))
data['cleaned_body_len'] = data["cleaned_body_text"].apply(lambda x: len(str(x)) - str(x).count(" "))
data['cleaned_char_count'] = data["cleaned_body_text"].apply(lambda x: len(str(x)))
data['word_count'] = data["cleaned_body_text"].apply(lambda x: count_words(str(x)))
data['cleaned_body_punct%'] = data["cleaned_body_text"].apply(lambda x: count_punct(str(x)))
data['body_nonstop_len'] = data["body_text_nostop"].apply(lambda x: len(str(x)) - str(x).count(" "))
data['body_stemm_len'] = data["body_text_stemmed"].apply(lambda x: len(str(x)) - str(x).count(" "))
data['body_lemm_len'] = data["body_text_lemmatized"].apply(lambda x: len(str(x)) - str(x).count(" "))
data['body_textlemm_nolongwords_len'] = data['body_textlemm_nolongwords'].apply(lambda x: len(str(x)) - str(x).count(" "))
data['proper_nouns_count'] = data["cleaned_body_text"].apply(lambda x: proper_nouns(str(x)))
data['nouns_count'] = data["cleaned_body_text"].apply(lambda x: nouns(str(x)))
data['persons'] = data["cleaned_body_text"].apply(lambda x: find_persons(str(x)))


# Reading Scores
# Import Textatistic
# TODO: Додати  лише для англійської мови!
data_en = data[data['language'] == 'en']

flesh_reading_scores = []
gunning_fog_scores = []

def reading_scores(cleaned_text):
    for article in cleaned_text:
        # Compute the readability scores
        try:
            readability_scores = Textatistic(article).scores
            flesch = readability_scores['flesch_score']
            gunning_fog = readability_scores['gunningfog_score']
            flesh_reading_scores.append(flesch)
            gunning_fog_scores.append(gunning_fog)
        except:
            continue

    return flesh_reading_scores, gunning_fog_scores


flesh_reading_scores, gunning_fog_scores = reading_scores(data["cleaned_body_text"])

data["flesh_reading_scores"] = pd.Series(flesh_reading_scores)
data["gunning_fog_scores"] = pd.Series(gunning_fog_scores)

# import datetime
# import pytz
# # timestamp for parsing date
# def create_timestamp():
#     # This timestamp is in UTC
#     my_ct = datetime.datetime.now(tz=pytz.UTC)
#     tz = pytz.timezone('Europe/Kiev')
#     # Now convert it to another timezone
#     new_ct = my_ct.astimezone(tz)
#     timestamp = new_ct.strftime("%d-%m-%Y_%H-%I")
#     return timestamp

# # create results in a csv file
# def create_csv(df, filename):
#     # file_timestamp = create_timestamp()
#     csv_file = df.to_csv(f'result_csv/{filename}_{create_timestamp()}.csv', index=False, encoding='utf-8')
#     return csv_file
create_csv(data, 'data_stat')
