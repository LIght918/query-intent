# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:47:06 2015

@author: haoyuxu
"""

import nltk

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
NROWS = 10000

class Utility:
    @staticmethod    
    def test_dropNan_UNKNOWN(df):
        df = df.dropna()
        df = df[df.apply(lambda x: ('CLASS=UNKNOWN' not in x['query'] and 'CLASS=TEST' not in x['query']),axis=1)]
        return df   

# flat list[tuple,tuple] into list[], and remove none from list    
    @staticmethod
    def wordpair_to_wordlist(word_series):
        wordlist = [element for tupl in word_series for element in tupl]
        wordlist_dropnone = [ word for word in wordlist if word!=None ]        
        return wordlist_dropnone
        
## calulate how many word in total
    @staticmethod
    def totalword(word_freq_df):
        wordlist = Utility.wordpair_to_wordlist(pd.Series.tolist(word_freq_df['wordpair']))        
        return wordlist

class Feature:
    
    wordvector_list = []
    wordvector_list_bigram = []
    df = pd.DataFrame()
    
    def __init__(self):
        print 'selecting features...'
    
    def loadData(self):

        df = pd.read_csv('D:\\CIKM\\train_trim.txt',nrows=NROWS)
        df = Utility.test_dropNan_UNKNOWN(df)        
        return df

###count_freq: 
###input, the dataframe 
###output, dataframe of the word frequency distribution

    def count_freq(self,df):
        word_freq = list()    
        
    ## iterate through dataframe, process each query, count the frequency
        for i in range(len(df)):    
            query = df.iloc[i,0]
    ## exclude NAN query
            if pd.isnull(query) == False:   
                queries = query.split('\t')[1:]
                #print queries
                for words in queries:
    ## words is not '-'
                    if words!='-':
                        bigram_words = nltk.bigrams(words.split(), pad_right=True, pad_left=True)
                        word_freq = word_freq + bigram_words
                            
        word_fd = nltk.FreqDist(word_freq)
        word_fd.tabulate()
        word_freq_df = pd.DataFrame()
        word_freq_df['wordpair'] = word_fd.keys()
        word_freq_df['count'] = word_fd.values()
        return word_freq_df

# get wordvector for each class include VIDEO”, “NOVEL”, “GAME”, “TRAVEL”, “LOTTERY”, “ZIPCODE
# length of each query is: video: 315, novel: 74, game: 74, travel: 48, lottery: 28, zipcode: 1
    def wordVec_class(self,df):
        video_df = df[df.apply(lambda x: 'video' in x['query'].lower(), axis=1)]
        novel_df = df[df.apply(lambda x: 'novel' in x['query'].lower(), axis=1)]
        game_df = df[df.apply(lambda x: 'game' in x['query'].lower(), axis=1)]
        travel_df = df[df.apply(lambda x: 'travel' in x['query'].lower(), axis=1)]
        lottery_df = df[df.apply(lambda x: 'lottery' in x['query'].lower(), axis=1)]
        zipcode_df = df[df.apply(lambda x: 'zipcode' in x['query'].lower(), axis=1)]
    ## get freq feature for each class, select pair that occur more than twice        
        video_fd = self.count_freq(video_df)
        novel_fd = self.count_freq(novel_df)
        game_fd = self.count_freq(game_df)
        travel_fd = self.count_freq(travel_df)
        lottery_fd = self.count_freq(lottery_df)
        zipcode_fd = self.count_freq(zipcode_df)
        
        return video_fd,novel_fd,game_fd,travel_fd,lottery_fd,zipcode_fd
    
    def construct_wordvector(self,video_fd,novel_fd,game_fd,travel_fd,lottery_fd,zipcode_fd):
    ## get each word vector for each class, zipcode>0 since only one record in zipcode class        
        wordvector_video = Utility.wordpair_to_wordlist(pd.Series.tolist(video_fd[video_fd['count']>0]['wordpair']))
        wordvector_novel = Utility.wordpair_to_wordlist(pd.Series.tolist(novel_fd[novel_fd['count']>0]['wordpair']))
        wordvector_game = Utility.wordpair_to_wordlist(pd.Series.tolist(game_fd[game_fd['count']>0]['wordpair']))
        wordvector_travel = Utility.wordpair_to_wordlist(pd.Series.tolist(travel_fd[travel_fd['count']>0]['wordpair']))
        wordvector_lottery = Utility.wordpair_to_wordlist(pd.Series.tolist(lottery_fd[lottery_fd['count']>0]['wordpair']))
        wordvector_zipcode = Utility.wordpair_to_wordlist(pd.Series.tolist(zipcode_fd[zipcode_fd['count']>0]['wordpair']))
        wordvector_list = wordvector_video+wordvector_novel+wordvector_game+wordvector_travel+wordvector_lottery+wordvector_zipcode
        return wordvector_list

## main function to word vector as feature
    def get_wordvector(self):
        df = self.loadData()
        self.df = df        
        video_fd,novel_fd,game_fd,travel_fd,lottery_fd,zipcode_fd = self.wordVec_class(df)
        self.wordvector_list = set(self.construct_wordvector(video_fd,novel_fd,game_fd,travel_fd,lottery_fd,zipcode_fd))
    
    def query_vectorization(self,query):
        query_vector = {word: 0 for word in self.wordvector_list}
                      
        queries = query.split('\t')[1:]        
        for one_query in queries:
            if one_query!='-':            
                words = one_query.split(' ')
                for word in words:
                    if query_vector.has_key(word):
                        query_vector[word] = query_vector[word]+1
## query_vector is the dictionary, instead return value of the dictionary
        return query_vector.values()
    
    def get_wordvector_bigram(self):
        video_fd,novel_fd,game_fd,travel_fd,lottery_fd,zipcode_fd = self.wordVec_class(self.df)
        self.wordvector_list_bigram = set(pd.Series.tolist(video_fd['wordpair'])+pd.Series.tolist(novel_fd['wordpair'])+pd.Series.tolist(game_fd['wordpair'])+pd.Series.tolist(travel_fd['wordpair'])+pd.Series.tolist(lottery_fd['wordpair'])+pd.Series.tolist(zipcode_fd['wordpair']))
        
    def query_vectorization_bigram(self,query):
        word_freq = list()         
        query_vector_bigram = {bigram_word: 0 for bigram_word in self.wordvector_list_bigram}
        queries = query.split('\t')[1:]        
        for words in queries:
            if words!='-':            
                bigram_words = nltk.bigrams(words.split(), pad_right=True, pad_left=True)
                word_freq = word_freq + bigram_words
## query_vector is the dictionary, instead return value of the dictionary
                          
        for word_bigram in word_freq:
            if query_vector_bigram.has_key(word_bigram):
                query_vector_bigram[word_bigram] = query_vector_bigram[word_bigram]+1
     
        return query_vector_bigram.values()
   
    def vectorization(self):    
        self.df['vector_query'] = self.df.apply(lambda x: self.query_vectorization(x['query']), axis=1)        
        self.df['vector_query_bigram'] = self.df.apply(lambda x: self.query_vectorization_bigram(x['query']), axis=1)        
        for i in range(len(self.wordvector_list)):
            self.df['unigram'+str(i)] = self.df.apply(lambda x: x['vector_query'][i],axis=1)
            print i, ': ',len(self.wordvector_list)
        
        for i in range(len(self.wordvector_list_bigram)):
            self.df['bigram'+str(i)] = self.df.apply(lambda x: x['vector_query_bigram'][i],axis=1)
            print i, ': ',len(self.wordvector_list_bigram)
            
        
## vector each query 
        

class Model:
    df = pd.DataFrame()
    wordvector_list = []
    wordvector_list_bigram = []
    trainData = pd.DataFrame() 
    trainLabel = pd.DataFrame()
    testData = pd.DataFrame()
    testLabel = pd.DataFrame()
    
    def __init__(self, df,wordvector_list,wordvector_list_bigram):
        self.df = df
        self.wordvector_list = wordvector_list
        self.wordvector_list_bigram = wordvector_list_bigram

    def label(self):
        self.df['label'] = self.df.apply(lambda x: 1 if 'VIDEO' in x['query'] else 0, axis=1)
    
## split data set for train and test
    def splitData(self):
        istrain = np.random.binomial(1,0.7,size=len(self.df))
        self.trainData = self.df.iloc[:,3:3+len(self.wordvector_list)][istrain==1]
        self.trainLabel = self.df['label'][istrain==1]
        self.testData = self.df.iloc[:,3:3+len(self.wordvector_list)][istrain==0]
        self.testLabel = self.df['label'][istrain==0]
    
    def splitData_bigram(self):
        istrain = np.random.binomial(1,0.7,size=len(self.df))
        self.trainData = self.df.iloc[:,3:3+len(self.wordvector_list)+len(self.wordvector_list_bigram)][istrain==1]
        self.trainLabel = self.df['label'][istrain==1]
        self.testData = self.df.iloc[:,3:3+len(self.wordvector_list)+len(self.wordvector_list_bigram)][istrain==0]
        self.testLabel = self.df['label'][istrain==0]            
    
    def fitSVM(self):
        clf = svm.SVC()  # class   
        clf.fit(self.trainData, self.trainLabel)  # training the svc model  
        pred = clf.predict(self.testData) # predict the target of testing samples
        print 'SVM precision: ', metrics.precision_score(self.testLabel, pred) 
        print 'SVM recall: ', metrics.recall_score(self.testLabel, pred) 
        
    def fitRF(self):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1,max_depth=3)
        clf.fit(self.trainData, self.trainLabel)  # training the svc model  
        pred = clf.predict(self.testData) # predict the target of testing samples
        print 'RF precision: ', metrics.precision_score(self.testLabel, pred) 
        print 'RF recall: ', metrics.recall_score(self.testLabel, pred) 
    
    def fitMNB(self):
        clf = MultinomialNB()
        clf.fit(self.trainData, self.trainLabel)  # training the svc model  
        pred = clf.predict(self.testData) # predict the target of testing samples
        print 'MultinomialNB precision: ', metrics.precision_score(self.testLabel, pred) 
        print 'MultinomialNB recall: ', metrics.recall_score(self.testLabel, pred) 
        
    def fitBNB(self):
        clf = BernoulliNB()
        clf.fit(self.trainData, self.trainLabel)  # training the svc model  
        pred = clf.predict(self.testData) # predict the target of testing samples
        print 'BernoulliNB precision: ', metrics.precision_score(self.testLabel, pred) 
        print 'BernoulliNB recall: ', metrics.recall_score(self.testLabel, pred) 
        
feature = Feature()
feature.get_wordvector()
feature.get_wordvector_bigram()
feature.vectorization()


model = Model(feature.df,feature.wordvector_list,feature.wordvector_list_bigram)
model.label()
model.splitData_bigram()
model.fitSVM()
model.fitRF()
model.fitMNB()
model.fitBNB()


'''
bigram added
SVM precision:  0.579268292683
SVM recall:  1.0
RF precision:  0.920792079208
RF recall:  0.978947368421
MultinomialNB precision:  0.910891089109
MultinomialNB recall:  0.968421052632
BernoulliNB precision:  0.77868852459
BernoulliNB recall:  1.0


without bigram
SVM precision:  0.583850931677
SVM recall:  1.0
RF precision:  0.805555555556
RF recall:  0.925531914894
MultinomialNB precision:  0.854368932039
MultinomialNB recall:  0.936170212766
BernoulliNB precision:  0.786324786325
BernoulliNB recall:  0.978723404255



'''


'''
##use pca to reduce dimension
pca = PCA(n_components=200, copy=True, whiten=False)
allData = model.df.iloc[:,3:3+len(model.wordvector_list)+len(model.wordvector_list_bigram)]
pca.fit(allData)
allData = pca.transform(allData)

istrain = np.random.binomial(1,0.7,size=len(model.df))
trainData_pca = allData[istrain==1]
trainLabel = model.df['label'][istrain==1]
testData_pca = allData[istrain==0]
testLabel = model.df['label'][istrain==0]   


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1,max_depth=3)
clf.fit(trainData_pca, trainLabel)  # training the svc model  
pred = clf.predict(testData_pca) # predict the target of testing samples
print 'RF precision: ', metrics.precision_score(testLabel, pred) 
print 'RF recall: ', metrics.recall_score(testLabel, pred) 



'''
