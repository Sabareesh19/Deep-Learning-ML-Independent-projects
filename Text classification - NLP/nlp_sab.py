# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:27:05 2018

@author: sabar
"""
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting is used to ignore the double quotes

#If we want to predict the text firstly we need to clean the text
#Cleaning the text
#Use stemming coz same set of words should not be repeated like love for loving, loved
import re 
import nltk
nltk.download('stopwords') #contains the list of all insignificant words to be removed
from nltk.corpus import stopwords #to use list of insignificant words use corpus
from nltk.stem.porter import PorterStemmer
corpus = []  #create a list named corpus
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #to specify in first remove to remove all chars except a-z,A-Z
    review = review.lower() #lower() will make all chars in review to lower case
    review = review.split()  #used to split an entire sentence into words
#Removing all insignificant words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  #now it contains list of irrelevant words
#Stemming is taking the root of the word
    review = ' '.join(review) #to join the different words of the review
    corpus.append(review) #need to append all cleaned review to the corpus
    
#Creating Bag of words model
#Import the tokenization model.
from sklearn.feature_extraction.text import CountVectorizer  #Countvectorizer can automatically clean the data
cv = CountVectorizer(max_features = 1500)  #max_features is to remove most of irrelavent words
X = cv.fit_transform(corpus).toarray()  #from matrix of values to array to create the sparse matrix,
#X is an independent variable which has 1500 dependent values
y = dataset.iloc[:, 1].values #creating the dependent variable

# Splitting the dataset into the Training set and Test set
#No need of feature extraction
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(54/200)*100