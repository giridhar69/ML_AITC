# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:39:14 2018

@author: C5232886
"""
# Importing prerequisite libraries
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd

# Declaring the documents we use in this example
documents= [' I love programming' , 'Programming also loves me']

# Stopwords
stopwordslist = nltk.corpus.stopwords.words('english')

# Normalization of text
def normalize_text(text):
    text=text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
    text = re.sub('@[^\s]+','', text)
    text = re.sub('#([^\s]+)', '', text)
    text = re.sub('[:;>?<=*+()&,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
    text = re.sub('[\d]','', text)
    text = text.replace(".", '')
    text = text.replace("'", '')
    text = text.replace("`", '')
    text = text.replace("'s", '')
    text = text.replace("/", ' ')
    text = text.replace("\"", ' ')
    text = text.replace("\\", '')
    #text =  re.sub(r"\b[a-z]\b", "", text)
    text=re.sub( '\s+', ' ', text).strip()
    return text

# Lemmmatizer
lemmatizer = WordNetLemmatizer()

# Stemming is also applied after the lemmatizing
docs = []
for doc in documents:
    text = normalize_text(doc)
    nl_text = ''
    for word in word_tokenize(text):
        if word not in stopwordslist:
            lemmatizer.lemmatize(word)
            nl_text += (PorterStemmer().stem(word))+' '
    docs.append(nl_text)

# printing the documents
print(docs)    

# printing the counts of the words in the docs
import collections
words=" ".join(docs).split() 
count= collections.Counter(words).most_common()
print(count)

#create a lexicon [features]
features=[c[0] for c in count]
print('The Lexicon is as follows:',features)

# Building the bag of words
import numpy as np
training_examples=[]
for doc in docs:
    doc_feature_values = np.zeros(len(features))
    for word in word_tokenize(doc):
        if word in features:
            index=features.index(word)
            doc_feature_values[index] +=1
    training_examples.append(doc_feature_values)

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(docs)
freqs = [(word, bow.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
results=sorted (freqs, key = lambda x: -x[1])
print('Count Vectorizer:')
print(results)

# To show as a table the occurance of the words
feature_names = vectorizer.get_feature_names()
corpus_index = [n for n in docs]
df = pd.DataFrame(bow.todense(), index=corpus_index, columns=feature_names)
print(df)

# TF-IDF (most important)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer1 = TfidfVectorizer(sublinear_tf=True, max_df=1.0)
bow1 = vectorizer1.fit_transform(docs)
freqs1 = [(word, bow1.getcol(idx).sum()) for word, idx in vectorizer1.vocabulary_.items()]
results1=sorted (freqs1, key = lambda x: -x[1])
print('TF-IDF Vectorizer:')
print(results1)

# To show as a table the TF-IDF calculated
feature_names = vectorizer1.get_feature_names()
corpus_index = [n for n in docs]
df = pd.DataFrame(bow1.todense(), index=corpus_index, columns=feature_names)
print(df)