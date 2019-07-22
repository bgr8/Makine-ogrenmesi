# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:20:24 2019

@author: Buğra
"""

import pandas as pd

# %% import twitter data
data = pd.read_csv(r"gender_classifier.csv", encoding = "latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis = 0,inplace = True)
data.gender = [ 1 if each == "female" else 0 for each in data.gender]

# %% cleaning data
# regular exp = RE
import re

first_description = data.description[4]
description = re.sub("[^a-zA-z]", " ", first_description) # ^bulma demek
description = description.lower()

# %% stopwords (irrelavent words) gereksiz kelimeler
import nltk
nltk.download("stopwords")
nltk.download("punkt") # word_tokenize için
from nltk.corpus import stopwords

#description = description.split()

# split yerine tokenizer
description = nltk.word_tokenize(description)

# split kullanırsak "shouldn't" gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize ile ayrılır

# %%
# gereksiz kelimeleri çıkar
description = [ word for word in description if not word in set(stopwords.words("english"))]

# %% 
# lemmatization loved => love

import nltk as nlp
nltk.download('wordnet')

lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description]

description = " ".join(description)

#%%
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

# %% bag of words
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak için kullanılan metot
max_features = 500 # en çok kullanılan kelimeden 500 tane seç

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("en sık kullanılan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))

# %%
y = data.iloc[:,0].values # male or female classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction
y_pred = nb.predict(x_test)

print("accuracy: ", nb.score(y_pred.reshape(-1,1), y_test))

































