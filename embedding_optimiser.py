'''
Created on 25 Jan 2016

@author: code copied from http://un-mindlab.blogspot.com.au/
multiplicative update rules from Seung and Lee
'''
import theano
import theano.tensor as T
import theano.sparse as Ts
import numpy as np
import time
import scipy.sparse as sp
import scipy
import sys
import itertools
from scipy.special import expit as sigmoid
import copy
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix
from sklearn.preprocessing import normalize
import pdb
from sklearn.preprocessing import MinMaxScaler 
import networkx as nx
from scipy.special import expit
import logging
import random
from scipy.sparse import lil_matrix
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def edgexplain_geolocate_iterative(X, G, train_ids, test_ids, id_index, label_slices, preserve_coef=0.9, iterations=10, alpha=10, C=0, node_order='random', keep_topK=10, edgexplain__scaler=False):
    if edgexplain__scaler:
        print 'edgexplain is so on!'
    else:
        print 'edgexplain is so off!'
    print 'alpha', alpha, 'preserver', preserve_coef, 'topK', keep_topK, 'edgexplain', edgexplain__scaler
    converged = False
    iter_num = 0
    ids = train_ids + test_ids
    train_ids_set = set(train_ids)
    logging.info("iterating with max_iter = " + str(iterations))
    X = X.tolil()
    while not converged:
        if node_order == 'random':
            random.shuffle(ids)
        logging.info( "iter: " + str(iter_num))
        for node in ids:
            node_index = id_index[node]
            neighbors_labeldist = lil_matrix((1, X.shape[1]))
            nbrs = [nbr for nbr in G[node]]
            ################
            
            nbr_indices = [id_index[n] for n in nbrs]
            if len(nbr_indices) == 0:
                continue
            #all neighbours
            nbrlabeldists = X[nbr_indices]
            #diagonal matrix for edge weight of each neighbor
            weights = lil_matrix((nbrlabeldists.shape[0], nbrlabeldists.shape[0]))
            if edgexplain__scaler:
                #edge weights are scaled according to EdgExplain scalar
                scales = nbrlabeldists.tocsr().dot(X[node_index].tocsr().transpose(copy=True))
                if iter_num == -1:
                    pdb.set_trace()
                if C == 0:
                    c = 0
                else:
                    #per user c
                    c = C[node, nbr_indices]
                scales = expit(-alpha * scales.toarray() - c)
                weights.setdiag(scales)
            else:
                #all edge weights are 1
                weights.setdiag(np.ones(nbrlabeldists.shape[0]))
            neighbors_labeldist = weights.dot(nbrlabeldists)
            neighbors_labeldist = lil_matrix(neighbors_labeldist.sum(axis=0) / weights.sum())

            if node in train_ids_set:
                new_labeldist = (preserve_coef * X[node_index] + (1-preserve_coef) * neighbors_labeldist).tolil()
            else:
                ignore_previous_labels = False
                if ignore_previous_labels or not edgexplain__scaler:
                    #ignore the i-1 labels and update to ith iteration labels
                    new_labeldist = neighbors_labeldist
                else:
                    #sum of the i-1 and i iterations' labels
                    new_labeldist = (X[node_index] + neighbors_labeldist).tolil()

            new_labeldist_normalized = None
            for label_slice in label_slices:
                start_index, end_index = label_slice
                slice = new_labeldist[0, start_index: end_index]
                if keep_topK > 0:
                    if keep_topK < 1:
                        keep_topK = min(1, int(keep_topK * slice.shape[1]))
                    sorted_indices = np.argsort(slice.toarray())
                    #topK_indices = sorted_indices[0, -keep_topK:]
                    zero_indices = sorted_indices[0, 0:-keep_topK]
                    slice[0,zero_indices] = 0
                slice = normalize(slice, norm='l1', axis=1, copy=False)
                if str(type(new_labeldist_normalized)) == '<type \'NoneType\'>':
                    new_labeldist_normalized = slice
                else:
                    new_labeldist_normalized = sp.hstack([new_labeldist_normalized, slice])
            new_labeldist = new_labeldist_normalized
            X[node_index] = new_labeldist.tolil()
        iter_num += 1
        if iter_num == iterations:
            converged = True
    X = X.tocsr()
    return X
        
        
def iterative_profiler(X, G, train_ids, test_ids, id_index, label_slices, preserve_coef=0.9, iterations=10, alpha=10, c=0, node_order='random', keep_topK=10, edgexplain__scaler=False):
    if edgexplain__scaler:
        print 'edgexplain is so on!'
    else:
        print 'edgexplain is so off!'
    print 'alpha', alpha, 'preserver', preserve_coef, 'c', c, 'topK', keep_topK, 'edgexplain', edgexplain__scaler
    converged = False
    iter_num = 0
    ids = train_ids + test_ids
    train_ids_set = set(train_ids)
    logging.info("iterating with max_iter = " + str(iterations))
    X = X.tolil()
    while not converged:
        if node_order == 'random':
            random.shuffle(ids)
        logging.info( "iter: " + str(iter_num))
        for node in ids:
            node_index = id_index[node]
            neighbors_labeldist = lil_matrix((1, X.shape[1]))
            nbrs = [nbr for nbr in G[node]]
            ################
            
            nbr_indices = [id_index[n] for n in nbrs]
            #all neighbours
            nbrlabeldists = X[nbr_indices]
            #diagonal matrix for edge weight of each neighbor
            weights = lil_matrix((nbrlabeldists.shape[0], nbrlabeldists.shape[0]))
            if edgexplain__scaler:
                #edge weights are scaled according to EdgExplain scalar
                scales = nbrlabeldists.tocsr().dot(X[node_index].tocsr().transpose(copy=True))
                if iter_num == -1:
                    pdb.set_trace()
                scales = expit(-alpha * scales.toarray() - c)
                weights.setdiag(scales)
            else:
                #all edge weights are 1
                weights.setdiag(np.ones(nbrlabeldists.shape[0]))
            neighbors_labeldist = weights.dot(nbrlabeldists)
            neighbors_labeldist = lil_matrix(neighbors_labeldist.sum(axis=0) / weights.sum())

            if node in train_ids_set:
                new_labeldist = (preserve_coef * X[node_index] + (1-preserve_coef) * neighbors_labeldist).tolil()
            else:
                ignore_previous_labels = False
                if ignore_previous_labels or not edgexplain__scaler:
                    #ignore the i-1 labels and update to ith iteration labels
                    new_labeldist = neighbors_labeldist
                else:
                    #sum of the i-1 and i iterations' labels
                    new_labeldist = (X[node_index] + neighbors_labeldist).tolil()

            new_labeldist_normalized = None
            for label_slice in label_slices:
                start_index, end_index = label_slice
                slice = new_labeldist[0, start_index: end_index]
                if keep_topK > 0:
                    if keep_topK < 1:
                        keep_topK = min(1, int(keep_topK * slice.shape[1]))
                    sorted_indices = np.argsort(slice.toarray())
                    #topK_indices = sorted_indices[0, -keep_topK:]
                    zero_indices = sorted_indices[0, 0:-keep_topK]
                    slice[0,zero_indices] = 0
                slice = normalize(slice, norm='l1', axis=1, copy=False)
                if str(type(new_labeldist_normalized)) == '<type \'NoneType\'>':
                    new_labeldist_normalized = slice
                else:
                    new_labeldist_normalized = sp.hstack([new_labeldist_normalized, slice])
            new_labeldist = new_labeldist_normalized
            X[node_index] = new_labeldist.tolil()
        iter_num += 1
        if iter_num == iterations:
            converged = True
    X = X.tocsr()
    return X
        
def edgexplain_profiler(X, training_indices, A, label_slices, iterations, learning_rate=0.1, alpha=10, C=0, lambda1=0.001, text_dimensions=None):
    '''
    alpha and c are edgeexplain variables
    A is the adjacancy matrix (e.g. word-word relations)
    X is the word embeddings already available.
    X_bar is the hopefully improved word embeddings.
    '''
    X = X.toarray()
    normalize_x = False
    if normalize_x:
        X_new = None
        for label_slice in label_slices:
            start_index, end_index = label_slice
            slice = np.exp(X[:, start_index: end_index])
            slice = normalize(slice, norm='l1', axis=1, copy=False)
            if X_new == None:
                X_new = slice
            else:
                X_new = np.hstack((X_new, slice))
        X = X_new
    
    print 'initializing X_bar with X...'
    X_bar = X.copy()
    print 'creating the shared variables for X, X_bar and A ...'
    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tX_bar = theano.shared(X_bar.astype(theano.config.floatX),name="X_bar")
    tA = theano.shared(A, name="A")
    
    print 'defining the cost functions and its gradient...'
    tEmbedding = T.sum((tX[training_indices]-tX_bar[training_indices])**2)
    #tEmbedding = T.sum((tX-tX_bar)**2)
    if sp.issparse(A):
        #tEdgexplain = lambda1 * Ts.sp_sum(Ts.structured_log(Ts.structured_sigmoid(Ts.structured_add(Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())), c))), sparse_grad=True)
        tEdgexplain = T.sum(T.log(T.nnet.sigmoid(C + Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())).toarray())))
        #tEdgexplain = T.sum(Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())).toarray())
    else:
        tEdgexplain = T.sum(T.log(T.nnet.sigmoid(C + alpha * A * T.dot(tX_bar, tX_bar.transpose()))))
    
    tCost = (1-lambda1) * tEmbedding - lambda1 * tEdgexplain 
    #tCost = tEdgexplain
    tGamma = T.scalar(name="learning_rate")
    # Gradient of the cost function with regards to tX_bar (the new embeddings)
    tgrad_X_bar = T.grad(cost=tCost, wrt=tX_bar) 
    
    train_embedding = theano.function(
            inputs=[tGamma],
            outputs=[],
            updates={tX_bar:tX_bar - tGamma * tgrad_X_bar},
            name="train_embedding")
    
    print 'training in ' + str(iterations) + ' iterations...'
    for i in range(0,iterations):
        print 'iter ' + str(i) + ' divergence from original embeddings: ', np.linalg.norm(tX.get_value()-tX_bar.get_value()), 'cost', tCost.eval()
        train_embedding(np.asarray(learning_rate,dtype=theano.config.floatX))
        #set possible inf or nan value to a large number or zero
        #tX_bar.set_value(np.nan_to_num(tX_bar.get_value()))
        if not text_dimensions:
            normalize_params = False
            if normalize_params:
                X_bar = tX_bar.get_value()
                X_bar_normalized = None
                for label_slice in label_slices:
                    start_index, end_index = label_slice
                    slice = np.exp(X_bar[:, start_index: end_index])
                    slice = normalize(slice, norm='l1', axis=1, copy=False)
                    if X_bar_normalized is None:
                        X_bar_normalized = slice
                    else:
                        X_bar_normalized = np.hstack((X_bar_normalized, slice))

                tX_bar.set_value(X_bar_normalized)
    normalize_params = True
    if normalize_params:
        X_bar = tX_bar.get_value()
        X_bar_normalized = None
        for label_slice in label_slices:
            start_index, end_index = label_slice
            slice = np.exp(X_bar[:, start_index: end_index])
            slice = normalize(slice, norm='l1', axis=1, copy=False)
            if str(type(X_bar_normalized)) == '<type \'NoneType\'>':
                X_bar_normalized = slice
            else:
                X_bar_normalized = np.hstack((X_bar_normalized, slice))

        tX_bar.set_value(X_bar_normalized)
    return csr_matrix(tX_bar.get_value())


def edgexplain_geolocate(X, training_indices, A, iterations, learning_rate=0.1, alpha=10, C=0, lambda1=0.001, text_dimensions=None):
    '''
    alpha and c are edgeexplain variables
    A is the adjacancy matrix (e.g. word-word relations)
    X is the word embeddings already available.
    X_bar is the hopefully improved word embeddings.
    '''
    #X = normalize(X, norm='l1', axis=1, copy=True)

    print 'initializing X_bar with X...'
    X_bar = X.copy()
    print 'creating the shared variables for X, X_bar and A ...'
    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tX_bar = theano.shared(X_bar.astype(theano.config.floatX),name="X_bar")
    tA = theano.shared(A, name="A")
    
    print 'defining the cost functions and its gradient...'
    tEmbedding = T.sum((tX[training_indices]-tX_bar[training_indices])**2)
    #tEmbedding = T.sum((tX-tX_bar)**2)
    if sp.issparse(A):
        #tEdgexplain = lambda1 * Ts.sp_sum(Ts.structured_log(Ts.structured_sigmoid(Ts.structured_add(Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())), c))), sparse_grad=True)
        tEdgexplain = T.sum(T.log(T.nnet.sigmoid(C + Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())).toarray())))
        #tEdgexplain = T.sum(Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())).toarray())
    else:
        tEdgexplain = T.sum(T.log(T.nnet.sigmoid(C + alpha * A * T.dot(tX_bar, tX_bar.transpose()))))
    
    tCost = (1-lambda1) * tEmbedding - lambda1 * tEdgexplain 
    #tCost = tEdgexplain
    tGamma = T.scalar(name="learning_rate")
    # Gradient of the cost function with regards to tX_bar (the new embeddings)
    tgrad_X_bar = T.grad(cost=tCost, wrt=tX_bar) 
    
    train_embedding = theano.function(
            inputs=[tGamma],
            outputs=[],
            updates={tX_bar:tX_bar - tGamma * tgrad_X_bar},
            name="train_embedding")
    
    print 'training in ' + str(iterations) + ' iterations...'
    for i in range(0,iterations):
        print 'iter ' + str(i) + ' divergence from original embeddings: ', np.linalg.norm(tX.get_value()-tX_bar.get_value()), 'cost', tCost.eval()
        train_embedding(np.asarray(learning_rate,dtype=theano.config.floatX))
        #set possible inf or nan value to a large number or zero
        #tX_bar.set_value(np.nan_to_num(tX_bar.get_value()))
        if not text_dimensions:
            normalize_params = False
            if normalize_params:
                print 'normalizing new embeddings...'
                normalized_X_bar = np.exp(tX_bar.get_value())
                normalized_X_bar = normalize(normalized_X_bar, norm='l1', axis=1, copy=True)
                #X_train = X[0: training_indices.shape[0], :]
                #X_eval = normalized_X_bar[training_indices.shape[0]:, :]
                #final_X = np.vstack((X_train, X_eval))
                tX_bar.set_value(normalized_X_bar)
        else:
            scaler = MinMaxScaler()
            X_bar = tX_bar.get_value()
            X_location = X_bar[:, 0: X_bar.shape[1] - text_dimensions]
            
            X_location_scaled = np.transpose(scaler.fit_transform(np.transpose(X_location)))
            X_location_normalized = normalize(X_location_scaled, norm='l1', axis=1, copy=True)
            #X_text_scaled = np.transpose(scaler.fit_transform(np.transpose(X_text)))
            #X_text_normalized = normalize(X_text_scaled, norm='l1', axis=1, copy=True)
            #keep the text dimensions unchanged
            X_text = X[:,(X.shape[1] - text_dimensions):]
            X_bar_normalized = np.hstack((X_location_normalized, X_text))
            #keep the train samples unchanged
            #X_train = X[0: training_indices.shape[0], :]
            #X_eval = X_bar_normalized[training_indices.shape[0]:, :]
            #final_X = np.vstack((X_train, X_eval))
            tX_bar.set_value(X_bar_normalized)
        #keep the training samples unchanged
        #X_bar = tX_bar.get_value()
        #X_bar[training_indices] = X[training_indices]
        #tX_bar.set_value(X_bar)
        #tX_bar[training_indices] = tX_copy[training_indices]
    normalized_X_bar = np.exp(tX_bar.get_value())
    normalized_X_bar = normalize(normalized_X_bar, norm='l1', axis=1, copy=True)
    tX_bar.set_value(normalized_X_bar)
        
    return tX_bar.get_value()


def edgexplain_retrofitting(X, A, iterations, learning_rate=0.1, alpha=10, c=0, lambda1=0.001):
    '''
    alpha and c are edgeexplain variables
    A is the adjacancy matrix (e.g. word-word relations)
    X is the word embeddings already available.
    X_bar is the hopefully improved word embeddings.
    '''
    

    print 'initializing X_bar with X...'
    X_bar = X.copy()
    
    print 'creating the shared variables for X, X_bar and A ...'
    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tX_bar = theano.shared(X_bar.astype(theano.config.floatX),name="X_bar")
    tA = theano.shared(A, name="A")
    
    print 'defining the cost functions and its gradient...'
    tEmbedding = T.sum((tX-tX_bar)**2)
    if sp.issparse(A):
        #tEdgexplain = lambda1 * Ts.sp_sum(Ts.structured_log(Ts.structured_sigmoid(Ts.structured_add(Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())), c))), sparse_grad=True)
        tEdgexplain = T.sum(T.log(T.nnet.sigmoid(c + Ts.basic.mul(tA, alpha * T.dot(tX_bar, tX_bar.transpose())).toarray())))
    else:
        tEdgexplain = T.sum(T.log(T.nnet.sigmoid(c + alpha * A * T.dot(tX_bar, tX_bar.transpose()))))
    
    tCost = (1-lambda1) * tEmbedding -  lambda1 * tEdgexplain 

    tGamma = T.scalar(name="learning_rate")
    # Gradient of the cost function with regards to tX_bar (the new embeddings)
    tgrad_X_bar = T.grad(cost=tCost, wrt=tX_bar) 
    
    train_embedding = theano.function(
            inputs=[tGamma],
            outputs=[],
            updates={tX_bar:tX_bar - tGamma * tgrad_X_bar},
            name="train_embedding")
    
    print 'training in ' + str(iterations) + ' iterations...'
    for i in range(0,iterations):
        print 'iter ' + str(i) + ' divergence from original embeddings: ', np.linalg.norm(tX.get_value()-tX_bar.get_value())
        train_embedding(np.asarray(learning_rate,dtype=theano.config.floatX))
        #set possible inf or nan value to a large number or zero
        tX_bar.set_value(np.nan_to_num(tX_bar.get_value()))
        
    return tX_bar.get_value()



def iterative_edgexplain_retrofitting(X, A, iterations, c_k=None, alpha=10, c=0):
    '''
    alpha and c are edgeexplain variables
    A is the adjacancy matrix (document-document relations)
    X is the word embeddings
    X_bar is the hopefully improved word embeddings
    '''
    if not c_k: 
        c_k = 1.0 / X.shape[1]
    X_new = copy.deepcopy(X)
    for i in range(iterations):
        print 'iter', i, 'divergence from original', np.linalg.norm(X - X_new)
        iter_gradients = alpha * np.dot(sigmoid(-c - alpha * A * np.dot(X_new, X_new.transpose())), X_new)
        X_new = X_new + c_k * iter_gradients 
    return X_new
        







def embedding_edgexplain(X, r, iterations, A, H=None, W=None, learning_rate=0.1, alpha=10, c=0):
    '''
    alpha and c are edgeexplain variables
    H and W are document-topic and topic-word matrices which should be inferred from input matrix X
    r is the embedding dimensionality (e.g. number of topics)
    A is the adjacancy matrix (document-document relations)
    '''
    rng = np.random
    n = X.shape[0]
    m = X.shape[1]
    #because we have more terms in the cost function, the model converges slower and needs more iterations.
    iterations *= 6
    #coefficients
    lambda1 = 0.001
    lambda2 = 0.001
    #note if lambda3 is high the model doesn't converge
    lambda3 = 0.0001
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)
    
    use_regularized_ridge_regression_for_h = False
    I = np.identity(r).astype(theano.config.floatX)
    tI = theano.shared(I, name="I")
    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tH = theano.shared(H,name="H")
    tW = theano.shared(W,name="W")
    tA = theano.shared(A, name="A")
    
    tL1 = T.sum(abs(tW))
    smooth_out_L1 = False
    if smooth_out_L1:
        epsilon = 0.1
        tL1 = T.sqrt(tW ** 2 + epsilon).sum()
    tL2 = T.sum(tH ** 2)
    '''
    Note: we don't need the sqrt in the cost function maximizing x^2 is similar to maximizing sqrt(x^2)... 
    but the function becomes very big and becomes inf if we don't do the sqrt.
    One possible solution is to divide it by a very big number to avoid inf.
    '''
    tEmbedding = T.sqrt(((tX-T.dot(tW,tH))**2).sum())
    #tEmbedding = ((tX-T.dot(tW,tH))**2).sum()
    tRegularizer = lambda1 * tL1 + lambda2 * tL2 
    #tEdgexplain = lambda3 * (T.log(1.0 / (1 + T.exp(-(c + alpha * tA * T.dot(tW, tW.transpose())))))).sum()
    tEdgexplain = lambda3 * (T.log(T.nnet.sigmoid(c + alpha * tA * T.dot(tW, tW.transpose())))).sum()
    #Note: Higher tEdgexplain should induce lower penalty
    tCost = tEmbedding - tEdgexplain + tRegularizer
    tGamma = T.scalar(name="learning_rate")
    tgrad_H, tgrad_W = T.grad(cost=tCost, wrt=[tH, tW]) 

    trainH = theano.function(
            inputs=[tGamma],
            outputs=[tCost],
            updates={tH:tH - tGamma * tgrad_H},
            name="trainH")
    trainHDirect = theano.function(
            inputs=[],
            outputs=[tCost],
            updates={tH:T.dot(T.dot( T.inv(T.dot(tW.T, tW) + lambda2 * tI ), tW.T), tX)},
            name="trainHDirect")                                   
                        
    trainW = theano.function(
            inputs=[tGamma],
            outputs=[tCost],
            updates={tW:tW - tGamma * tgrad_W},
            name="trainW")

    for i in range(0,iterations):
        if use_regularized_ridge_regression_for_h:
            tCostH = trainHDirect()
        else:
            tCostH = trainH(np.asarray(learning_rate,dtype=theano.config.floatX))
        tCostW = trainW(np.asarray(learning_rate,dtype=theano.config.floatX));
        print 'iter ' + str(i) + ':', np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        

    return tW.get_value(),tH.get_value()


if __name__=="__main__":
    print "USAGE : embedding_optimiser.py <matrixDim> <latentFactors> <iter>"
    print 'input matrix X is assumed to be a square for simplicity, the algorithms work with any type of input matrix.'
    
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    it = int(sys.argv[3])
    rng = np.random
    #topic-word
    Hi = rng.random((r,n)).astype(theano.config.floatX)
    #embeddings (document-topic)
    Wi = rng.random((n,r)).astype(theano.config.floatX)
    #input matrix  (document-word)
    X = rng.random((n,n)).astype(theano.config.floatX)
    #adjacancy matrix (document-document)
    A = rng.random((n,n)).astype(theano.config.floatX)
    print " --- "
    t0 = time.time()
    W,H = embedding_edgexplain(X, r, it, A, Hi, Wi, learning_rate=0.1, alpha=10, c=0)
    t1 = time.time()
    print "Time taken by CPU : ", t1-t0
