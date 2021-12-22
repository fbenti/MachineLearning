#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exercise 3.1.4
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# We'll use a widely used stemmer based:
# Porter, M. “An algorithm for suffix stripping.” Program 14.3 (1980): 130-137.
# The stemmer is implemented in the most used natural language processing
# package in Python, "Natural Langauge Toolkit" (NLTK):
from toolbox_02450.similarity import similarity
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()  

# FISRT SENTENCE

raw_file = "the bag of words representation should not give you a hard time"
corpus = raw_file.split('\n')
corpus = list(filter(None, corpus))

vectorizer.fit(corpus)
attributeNames = vectorizer.get_feature_names()
X = vectorizer.transform(corpus)
X = X.toarray()

# SECOND SENTENCE

q2 = vectorizer.transform(["remember the representation should be a vector"])
q2 = np.asarray(q2.toarray())

sim = similarity(X, q2, 'cos')

print("\n--- Similarity : {}".format(sim))

# # Display the result
# print('Document-term matrix analysis')
# print()
# print('Corpus (5 documents/sentences):')
# print(np.asmatrix(corpus))
# print()


# # To automatically obtain the bag of words representation, we use sklearn's
# # feature_extraction.text module, which has a function CountVectorizer.
# # We make a CounterVectorizer:
# vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b')   
# # The token pattern is a regular expression (marked by the r), which ensures 
# # that the vectorizer ignores digit/non-word tokens - in this case, it ensures 
# # the 10 in the last document is not recognized as a token. It's not important
# # that you should understand it the regexp.

# # The object vectorizer can now be used to first 'fit' the vectorizer to the
# # corpus, and the subsequently transform the data. We start by fitting:
# vectorizer.fit(corpus)
# # The vectorizer has now determined the unique terms (or tokens) in the corpus
# # and we can extract them using:
# attributeNames = vectorizer.get_feature_names()
# print('Found terms:')
# print(attributeNames)
# print()

# # The next step is to count how many times each term is found in each document,
# # which we do using the transform function:
# X = vectorizer.transform(corpus)
# N,M = X.shape
# print('Number of documents (data objects, N):\t %i' % N)
# print('Number of terms (attributes, M):\t %i' % M )
# print()
# print('Document-term matrix:')
# print(X.toarray())
# print()