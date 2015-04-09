# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:50:52 2015

@author: haoyuxu
"""
import pandas as pd

wordvector_list = [1,2,3,4,5]

def query_vectorization(query):
    query_vector = {word: 0 for word in wordvector_list}
                  
    queries = query.split('\t')[1:]        
    for one_query in queries:
        if one_query!='-':            
            words = one_query.split(' ')
            for word in words:
                if query_vector.has_key(word):
                    query_vector[word] = query_vector[word]+1
## query_vector is the dictionary, instead return value of the dictionary
    query_vector['query']=query
    return query_vector

d = {'query' : ['CLASS=VIDEO\t0933873 1116371\t-', 'CLASS=VIDEO\t0933353 13616371\t-']}
df = pd.DataFrame(d)

#print pd.DataFrame(query_vectorization('CLASS=VIDEO\t0933873 1116371\t-'))

df['vector'] = df['query'].apply(lambda x: pd.DataFrame(query_vectorization(x),index=[1]))