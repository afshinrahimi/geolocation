'''
Created on 27 Feb 2016
utility functions used in geolocation
@author: af
'''
import lda
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse.lil import lil_matrix
import scipy as sp
from sklearn.preprocessing.data import StandardScaler
from IPython.core.debugger import Tracer
import pickle
def topicModel(X_train, X_eval, components=20, vocab=None):
    '''
    given a list of texts, return the topic distribution of documents
    '''
    lda_model = lda.LDA(n_topics=components, n_iter=1500, random_state=1)
    X_train = lda_model.fit_transform(X_train)
    if vocab:
        topic_word = lda_model.topic_word_ 
        doc_topic = lda_model.doc_topic_
        n_top_words = 20
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            try:
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
            except:
                pass
    X_eval = lda_model.transform(X_eval, max_iter=500)
    return X_train, X_eval

def vectorize(train_texts, eval_texts, data_encoding, binary=False):
    if binary:
        vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=True, sublinear_tf=True, min_df=30, max_df=0.01, ngram_range=(1, 1), stop_words='english', vocabulary=None, encoding=data_encoding)
    else:
        vectorizer = CountVectorizer(stop_words='english', max_df=0.2, min_df=10, encoding=data_encoding)
    X_train = vectorizer.fit_transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)
    feature_names = vectorizer.get_feature_names()
    print feature_names[0:100], feature_names[-100:]
    return X_train, X_eval, vectorizer.get_feature_names()

def smoothness_terms(texts, locations, distance_function):
    #LP Zhu exact solution
    pass

def text_distance_correlation(texts, locations, W, distance_function, data_encoding):
    scaler = StandardScaler(copy=False)
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=True, sublinear_tf=True, min_df=10, max_df=0.2, ngram_range=(1, 1), stop_words='english', vocabulary=None, encoding=data_encoding)
    X_text = vectorizer.fit_transform(texts)
    X = X_text.dot(X_text.transpose(copy=True))
    X = X.toarray()
    X = X * W
    #X = scaler.fit_transform(pairwise_similarity)
    L = lil_matrix((len(locations), len(locations)))
    for i in range(len(locations)):
        for j in range((len(locations))):
            lat1, lon1 = locations[i]
            lat2, lon2 = locations[j]
            if W[i, j] != 0:
                L[i, j] = distance_function(lat1, lon1, lat2, lon2)
    X = np.reshape(X, X.shape[0] * X.shape[1]).tolist()
    L = np.reshape(L.toarray(), L.shape[0] * L.shape[1]).tolist()
    W = np.reshape(W, W.shape[0] * W.shape[1]).tolist()
    X = [X[i] for i in range(len(X)) if W[i] != 0]
    L = [L[i] for i in range(len(L)) if W[i] != 0]
    with open('X-L.pkl', 'wb') as outf:
        pickle.dump((X, L), outf)
    Tracer()()
    