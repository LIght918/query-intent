# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:22:44 2015

@author: haoyuxu

topic include: 
VIDEO”, “NOVEL”, “GAME”, “TRAVEL”, “LOTTERY”, “ZIPCODE

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NROWS=3000

df = pd.read_csv('D:\\CIKM\\train_trim.txt')

print 'loading training data...'

df['is_train'] = np.random.uniform(0,1,df.shape[0])<=0.75
train,test = df[df['is_train']==True], df[df['is_train']==False]

def queryFreq(querystr, queryCount):
    if pd.isnull(querystr) == False:

## split query to get word and topic        
        queries = querystr.split('\t')[1:-1]
        queryclass = querystr.split('\t')[0].split(' | ')
## get rid of class=TEST and CLASS=UNKNOW        
        if ('CLASS=TEST' not in queryclass and 'CLASS=UNKNOWN' not in queryclass):
## count the word occurance and the occurance in each topic
        
            topic_dic = {'CLASS=GAME':0,'CLASS=VIDEO':0,'CLASS=NOVEL':0,'CLASS=TRAVEL':0,'CLASS=LOTTERY':0,'CLASS=ZIPCODE':0,'CLASS=UNKNOWN':0}
            for query in queries:
                one_query = str(query).split(' ')
                for word in one_query:
                    if queryCount.has_key(word) == False:                
                        queryCount[word] = topic_dic
                    
                    for topic_class in queryclass:
                        if topic_class in topic_dic.keys():                    
                            queryCount[word][topic_class] = queryCount[word][topic_class] + 1
                        else:
                            queryCount[word]['CLASS=UNKNOWN'] = queryCount[word]['CLASS=UNKNOWN'] + 1
    return queryCount


def queryCount_dataFrame(df):
    queryCount = {}
    
    for i in range(1,len(df)-1): 
        query = df.iloc[i,0]
        queryCount = queryFreq(query, queryCount)
        
    
    names = ['word','CLASS=GAME','CLASS=LOTTERY','CLASS=NOVEL','CLASS=TRAVEL','CLASS=UNKNOWN','CLASS=VIDEO','CLASS=ZIPCODE']
    queryCount_df = pd.DataFrame(np.zeros((len(queryCount),len(names))))
    queryCount_df.columns=names
    queryCount_df['word'] = queryCount.keys()
    queryCount_df[names[1:]] = pd.DataFrame(queryCount.values())
    return queryCount_df



def querCount_percent(queryCount_df):    
    queryCount_df['sum'] = queryCount_df['CLASS=GAME']+queryCount_df['CLASS=LOTTERY']+queryCount_df['CLASS=NOVEL']+queryCount_df['CLASS=TRAVEL']+queryCount_df['CLASS=VIDEO']+queryCount_df['CLASS=ZIPCODE']+queryCount_df['CLASS=UNKNOWN']
    queryCount_df['game'] = queryCount_df['CLASS=GAME']*1.0/queryCount_df['sum']
    queryCount_df['lottery'] = queryCount_df['CLASS=LOTTERY']*1.0/queryCount_df['sum']
    queryCount_df['novel'] = queryCount_df['CLASS=NOVEL']*1.0/queryCount_df['sum']
    queryCount_df['travel'] = queryCount_df['CLASS=TRAVEL']*1.0/queryCount_df['sum']
    queryCount_df['video'] = queryCount_df['CLASS=VIDEO']*1.0/queryCount_df['sum']
    queryCount_df['zipcode'] = queryCount_df['CLASS=ZIPCODE']*1.0/queryCount_df['sum']
    return queryCount_df

print 'train words...'
queryCount_df = queryCount_dataFrame(train)
queryCount_df = querCount_percent(queryCount_df)

def test_dropNan_UNKNOWN(df):
    df = df.dropna()
    df = df[df.apply(lambda x: ('CLASS=UNKNOWN' not in x['query'] and 'CLASS=TEST' not in x['query']),axis=1)]
    return df


def perdict_test(querystr):
    class_game_prob = 0
    class_lottery_prob = 0
    class_novel_prob = 0
    class_travel_prob = 0
    class_video_prob = 0
    class_zipcode_prob = 0    
    queries = querystr.split('\t')[1]
    queryclass = querystr.split('\t')[0].split(' | ')

    words = queries.split(' ')
    for word in words:
## calculate probability for each class       
        if word in list(queryCount_df['word']):
            game_prob = (queryCount_df.loc[queryCount_df['word']==word,'game'].iloc[0])
            lottery_prob = (queryCount_df.loc[queryCount_df['word']==word,'lottery'].iloc[0])
            novel_prob = (queryCount_df.loc[queryCount_df['word']==word,'novel'].iloc[0])
            travel_prob = (queryCount_df.loc[queryCount_df['word']==word,'travel'].iloc[0])
            video_prob = (queryCount_df.loc[queryCount_df['word']==word,'video'].iloc[0])
            zipcode_prob = (queryCount_df.loc[queryCount_df['word']==word,'zipcode'].iloc[0])
            class_game_prob    +=  game_prob    if game_prob!=0      else 0.001
            class_lottery_prob +=  lottery_prob if lottery_prob!=0   else 0.001
            class_novel_prob   +=  novel_prob   if novel_prob!=0     else 0.001
            class_travel_prob  +=  travel_prob  if travel_prob!=0    else 0.001
            class_video_prob   +=  video_prob   if video_prob!=0     else 0.001
            class_zipcode_prob +=  zipcode_prob if zipcode_prob!=0   else 0.001
            
    predict_list = [class_game_prob,class_lottery_prob,class_novel_prob,class_travel_prob,class_video_prob,class_zipcode_prob]
    return predict_list
## split query and count probiblity in each class


def predict_prob(test):
# dataframe to store probablity
    predict_ret = pd.DataFrame()
    test_dropna = test_dropNan_UNKNOWN(test) 

# iterate through dataframe and predict probability to each class
    for i in range(len(test_dropna)):
        ## from queries, get queryclass and querystr
        querystr = test_dropna.iloc[i,0]
        predict_list = perdict_test(querystr)
        predict_ret = pd.concat([predict_ret,pd.DataFrame(predict_list).T],axis=0)
    
    predict_ret.index = test_dropna.index
    predict_ret.columns = ['game','lottery','novel','travel','video','zipcode']
    test_dropna.columns = ['query','istrain']
    ret = pd.concat([test_dropna,predict_ret],axis=1)
    ret['ture_class'] = ret.apply(lambda x: x['query'].split('\t')[0],axis=1)
    
# function to find class with max probability    
    def findmax_prob(game,lottery,novel,travel,video,zipcode):
        max_prob = 0  
        max_class = ''
        max_prob,max_class = [game,'game'] if game>lottery else [lottery,'lottery']
        max_prob,max_class = [max_prob,max_class] if max_prob>novel else [novel,'novel']
        max_prob,max_class = [max_prob,max_class] if max_prob>travel else [travel,'travel']
        max_prob,max_class = [max_prob,max_class] if max_prob>video else [video,'video']
        max_prob,max_class = [max_prob,max_class] if max_prob>zipcode else [zipcode,'zipcode'] 
        return max_class
    
    ret['predict_class'] = ret.apply(lambda x: findmax_prob(x['game'],x['lottery'],x['novel'],x['travel'],x['video'],x['zipcode']) ,axis=1)
    return ret
    
ret = predict_prob(test)

def accuracy(ret):
    ret['score'] = ret.apply(lambda x: 1 if (x['predict_class'] in x['ture_class'].lower() or x['ture_class']=='CLASS=OTHER') else 0,axis=1)
    return sum(ret['score'])*1.0/len(ret)

accuracy = accuracy(ret)
