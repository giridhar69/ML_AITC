# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:07:42 2018

@author: C5232886
"""
#variables dictionary
'''
string -> sentence_1, sentence_1
list   -> words_1, words_2, lemmatized, stemmed, stopwordslist
          stopwordsremoved, normalized
'''

# Importing prerequisite libraries
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Tonkenizing of the sentences to derive words
sentence_1 = 'I love programming'
words_1 = word_tokenize(sentence_1)
sentence_2 = 'I love programming and programming also loves me'
words_2 = word_tokenize(sentence_2)

# Lemmatization is used to remove inflectional words like love and loves.
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in words_2]

# Stemming is used to normalize words like 'programming' into 'program'
stemmed = [PorterStemmer().stem(word) for word in lemmatized]

# Removing the stop words
stopwordslist = nltk.corpus.stopwords.words('english')
print('Number of Stop words in English language',len(stopwordslist))
stopwordsremoved = []
for word in stemmed:
    if word.lower() not in stopwordslist:
        stopwordsremoved.append(word)
'''
The sentence 'I love programming and programming also loves me' gets condensed into
program,also, love words

Still we need to remove symbols, links, hashtags, numbers, spaces etc., if any
We call this data normalization
'''
# Declaring the normalization function
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
normalized = []
for word in list(set(stopwordsremoved)):
    normalized.append(normalize_text(word))
print('Applied Lemmatization, Stemming, Stopword removal, lowercase conversion, normalization:',set(stopwordsremoved))




