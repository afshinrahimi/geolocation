'''
Created on 27 Feb 2016
utility functions used in geolocation
@author: af
'''
import lda
import numpy as np

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
    X_eval = lda_model.transform(X_eval, max_iter=100)
    return X_train, X_eval
