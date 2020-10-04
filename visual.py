from .jobscraper import create_timestamp
from .stat import data

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import en_core_web_md
nlp = en_core_web_md.load()
from spacy import displacy
stopwords = stopwords.words('english')
from stop_words import get_stop_words
stop_words_en = get_stop_words('en')
stop_words_ru = get_stop_words('russian')
# for using Google Colab notebook upload file and read it with pd.read_csv from your catalog
# stopwords_ua = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/stopwords_ua.txt", header=None, names=['stopwords'])
stopwords_ua = pd.read_csv("stopwords_ua.txt", header=None, names=['stopwords'])
stop_words_ua = list(stopwords_ua.stopwords)
from textblob import TextBlob


# Visualization

# Text length, word count, number of tokens, nostopwors length
# Drop all rows that are missing values"""
data['body_len'] = data['body_len'].dropna()

# Drop all rows that are missing 'driver_gender'
data.dropna(subset=['body_len'], inplace=True)

# Count the number of missing values in each column (again)
print(data.isnull().sum())

# Examine the shape of the DataFrame
print(data.shape)

plt.plot(data['cleaned_body_len'], color='g', marker='.', label='cleaned_body_len')
plt.title("Cleaned Body Length Distribution")
plt.grid(True)
plt.xticks(np.arange(0, 37, 1.0))
plt.show()

plt.plot(data.index, data['word_count'], marker='o', label='word_count')
plt.title("Word Count Distribution")
plt.grid(True)
plt.show()

plt.plot(data.index, data['body_textlemm_nolongwords_len'], marker='.', label='textlemm_nolongwords_len')
plt.title("Text Length w/o long words")
plt.grid(True)
plt.show()

plt.boxplot(data['body_textlemm_nolongwords_len'], showfliers=False)
plt.title("Words count w/o long words")
plt.show()
plt.savefig('plots/figure_2_len.png')

fig = plt.figure(figsize=(4, 8))
plt.boxplot(data['cleaned_body_len'], showfliers=False)
plt.title("Cleaned Body Length")
plt.show()
plt.savefig('plots/figure_3_len.png')

# Boxplot
def boxplot_default():
    fig = plt.figure(figsize=(15, 8))

    body_length = data[['body_len', 'cleaned_body_len', 'cleaned_char_count', 'word_count',
                        'body_textlemm_nolongwords_len']]
    body_length.boxplot(showfliers=False, patch_artist=True)

    plt.ylabel('Length')
    plt.title('Text and Words Length Distribution')
    plt.tick_params(labelrotation=45)
    fig.set_size_inches([18, 10])
    fig.savefig('plots/figure_4_boxplot.png')
    plt.show()

boxplot_default()

# Barplot

fig, ax = plt.subplots(figsize=(10, 10))
a = data['site'].value_counts()
species = a.index
count = a.values
sns.barplot(y=species, x=count, label='value_counts')
plt.show()
plt.savefig('plots/figure_5_plot.png')

# Lineplot

sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(data=data.cleaned_body_len)
plt.show()

# Wordcloud

# Use a cleaned body text
for i, s in enumerate(data['cleaned_body_text']):
    # Create and generate a word cloud image
    try:
        my_cloud = WordCloud(background_color='white', stopwords=stopwords).generate(str(s))
    except:
        print('Error has occured')
        continue


    # Display the generated wordcloud image
    plt.imshow(my_cloud, interpolation='bilinear')
    plt.axis("off")

    # Don't forget to show the final image
    print('---------------------------------')
    fig.set_size_inches([18, 10])
    plt.savefig(f'wordcloud/wordcloud_{i}.png')
    plt.show()

# Body text without stopwords
for s in data['body_text_nostop']:
    ss = ' '.join(s)
    try:
        my_cloud = WordCloud(background_color='white').generate(str(ss))
    except:
        print('Error has occured')
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(my_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Body text with lemmatizing and no stopwords
for s in data.body_textlemm_nolongwords:
    # Create and generate a word cloud image
    ss = ' '.join(s)
    try:
        my_cloud = WordCloud(background_color='white', stopwords=stopwords).generate(str(ss))
    except:
        print('Error has occured')
        continue
    # Display the generated wordcloud image
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(my_cloud, interpolation='bilinear')
    plt.axis("off")

    # Don't forget to show the final image

    plt.show()
    print('---------------------------------')

# Reading Scores (Flesh and Gunning)

plt.scatter(data["flesh_reading_scores"], data["gunning_fog_scores"])
plt.title('Reading Scores')
plt.xlabel('flesh_reading_scores')
plt.ylabel('gunning_fog_scores')
plt.show()

# Scatterplot of proper nouns and nouns

fig, ax = plt.subplots(figsize=(15, 8))
plt.scatter(data.index, data['proper_nouns_count'], label='proper_nouns_count', color='brown')
plt.legend()
fig.set_size_inches([18, 10])
fig.savefig('plots/figure_10.png')
plt.show()

fig, ax = plt.subplots(figsize=(15, 8))
plt.scatter(data.index, data['proper_nouns_count'], label='nouns_count', color='blue')
plt.legend()
fig.set_size_inches([18, 10])
fig.savefig('plots/figure_11.png')
plt.show()

# TODO: Clean persons from fuzzy words and symbols

for i, article in enumerate(data.cleaned_body_text):
    article = nlp(article)
    len(article.ents)
    labels = [x.label_ for x in article.ents]
    Counter(labels)
    items = [x.text for x in article.ents]
    Counter(items).most_common(10)
    sentences = [x for x in article.sents]
    html = displacy.render(nlp(str(article)), style='ent', jupyter=False, options={'distance': 120})

    # Write HTML  to file.html
    with open(f"file_{create_timestamp()}_{i}.html", "w") as file:
        file.write(html)


# compute sentiment scores (polarity) and labels
sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in data.cleaned_body_text]
sentiment_category_tb = ['positive' if score > 0
                             else 'negative' if score < 0
                                 else 'neutral'
                                     for score in sentiment_scores_tb]

# sentiment statistics per news category
df = pd.DataFrame([list(data['site']), sentiment_scores_tb, sentiment_category_tb]).T
df.columns = ['site_category', 'sentiment_score', 'sentiment_category']
df['sentiment_score'] = df.sentiment_score.astype('float')
df.groupby(by=['site_category']).describe()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
sns.stripplot(x='site_category', y="sentiment_score",
                   hue='site_category', data=df, ax=ax1)
f.savefig('plots/sentiment_stripplot.png')
sns.boxplot(x='site_category', y="sentiment_score",
                 hue='site_category', data=df, palette="Set2", ax=ax2)
f.suptitle('Visualizing Site Sentiment', fontsize=14)
f.savefig('plots/sentiment_boxplot.png')
plt.show()

fc = sns.catplot(x="site_category", hue="sentiment_category",
                    data=df, kind="count",
                    palette={"negative": "#FE2020",
                             "positive": "#BADD07",
                             "neutral": "#68BFF5"})
fc.savefig('plots/sentiment_catplot.png')