#!/usr/bin/python

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

doc_list = ['if you take the blue pill, the story ends',
	    'if you take the red pill, you stay in Wonderland',
	    'if you take the red pill, I show you how deep the rabbit hole goes']

tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(doc_list)

similarity_simple_pair = cosine_similarity(feature_vect_simple, feature_vect_simple)

print(similarity_simple_pair)
print()

value=[]
n = len(doc_list)
i =1 
for j in range(n):
	value.append(similarity_simple_pair[i,j])
value.sort(reverse=True)
print(value)
	
