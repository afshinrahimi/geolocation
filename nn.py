'''

Created on 22 Apr 2016

@author: af
'''

import pdb
import numpy as np
import sys
from os import path
import scipy as sp
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
import params
import geolocate
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict
from _collections import defaultdict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

class SparseInputDenseLayer(DenseLayer):
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        activation = S.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
class SparseInputDropoutLayer(DropoutLayer):
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)


class MLP():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 init_parameters=None, 
                 complete_prob=False, 
                 add_hidden=True, 
                 regul_coefs=[5e-5, 5e-5], 
                 save_results=False, 
                 hidden_layer_size=None, 
                 drop_out=False, 
                 drop_out_coefs=[0.5, 0.5]):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_parameters = init_parameters
        self.complete_prob = complete_prob
        self.add_hidden = add_hidden
        self.regul_ceofs = regul_coefs
        self.save_results = save_results
        self.hidden_layer_size = hidden_layer_size
        self.drop_out = drop_out
        self.drop_out_coefs = drop_out_coefs

    def fit(self, X_train, Y_train, X_dev, Y_dev):
        logging.info('building the network...' + ' hidden:' + str(self.add_hidden))
        in_size = X_train.shape[1]
        best_params = None
        best_val_acc = 0.0
        drop_out_hid, drop_out_in = self.drop_out_coefs
        if self.complete_prob:
            out_size = Y_train.shape[1]
        else:
            out_size = len(set(Y_train.tolist()))
        
        if self.hidden_layer_size:
            pass
        else:
            hidden_layer_size = min(5 * out_size, int(in_size / 20))
        logging.info('input layer size: %d, hidden layer size: %d, output layer size: %d'  %(X_train.shape[0], hidden_layer_size, out_size))
        # Prepare Theano variables for inputs and targets
        if not sp.sparse.issparse(X_train):
            logging.info('input matrix is not sparse!')
            self.X_sym = T.matrix()
        else:
            self.X_sym = S.csr_matrix(name='inputs', dtype='float32')
        
        if self.complete_prob:
            self.y_sym = T.matrix()
        else:
            self.y_sym = T.ivector()    
        
        l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                         input_var=self.X_sym)
        
        if self.drop_out:
            l_in = lasagne.layers.dropout(l_in, p=drop_out_in)
    
        if self.add_hidden:
            if not sp.sparse.issparse(X_train):
                l_hid1 = lasagne.layers.DenseLayer(
                    l_in, num_units=hidden_layer_size,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())
            else:
                l_hid1 = SparseInputDenseLayer(
                    l_in, num_units=hidden_layer_size,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())
            if self.drop_out:
                self.l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
            
            self.l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=out_size,
            nonlinearity=lasagne.nonlinearities.softmax)
        else:
            if not sp.sparse.issparse(X_train):
                self.l_out = lasagne.layers.DenseLayer(
                    l_in, num_units=out_size,
                    nonlinearity=lasagne.nonlinearities.softmax)
                if self.drop_out:
                    l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
            else:
                self.l_out = SparseInputDenseLayer(
                    l_in, num_units=out_size,
                    nonlinearity=lasagne.nonlinearities.softmax)
                if self.drop_out:
                    l_hid1 = SparseInputDropoutLayer(l_hid1, drop_out_hid)
        
        
    
        if self.add_hidden:
            self.embedding = lasagne.layers.get_output(l_hid1, self.X_sym, deterministic=True)
            self.f_get_embeddings = theano.function([self.X_sym], self.embedding)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        pred = self.output.argmax(-1)
        loss = lasagne.objectives.categorical_crossentropy(self.output, self.y_sym)
        #loss = lasagne.objectives.multiclass_hinge_loss(output, y_sym)
        loss = loss.mean()
        
        
        l1_share_out = 0.5
        l1_share_hid = 0.5
        regul_coef_out, regul_coef_hid = self.regul_coefs
        logging.info('regul coefficient for output and hidden layers are ' + str(self.regul_coefs))
        l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
        l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)
        if self.add_hidden:
            l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef_hid * l1_share_hid
            l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef_hid * (1-l1_share_hid)
        loss = loss + l1_penalty + l2_penalty
    
        if self.complete_prob:
            self.y_sym_one_hot = self.y_sym.argmax(-1)
            self.acc = T.mean(T.eq(self.pred, self.y_sym_one_hot))
        else:
            acc = T.mean(T.eq(pred, self.y_sym))
        if self.init_parameters:
            lasagne.layers.set_all_param_values(self.l_out, self.init_parameters)
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        
        #print(params)
        #updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.01, momentum=0.9)
        #updates = lasagne.updates.sgd(loss, parameters, learning_rate=0.01)
        #updates = lasagne.updates.adagrad(loss, parameters, learning_rate=0.1, epsilon=1e-6)
        #updates = lasagne.updates.adadelta(loss, parameters, learning_rate=0.1, rho=0.95, epsilon=1e-6)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        self.f_train = theano.function([self.X_sym, self.y_sym], [loss, acc], updates=updates)
        self.f_val = theano.function([self.X_sym, self.y_sym], [loss, acc])
        self.f_predict = theano.function([self.X_sym], pred)
        self.f_predict_proba = theano.function([self.X_sym], self.output)
        
        
        X_train = X_train.astype('float32')
        X_dev = X_dev.astype('float32')
    
        if self.complete_prob:
            Y_train = Y_train.astype('float32')
            Y_dev = Y_dev.astype('float32')
        else:
            Y_train = Y_train.astype('int32')
            Y_dev = Y_dev.astype('int32')
    
        logging.info('training (n_epochs, batch_size) = (' + str(self.n_epochs) + ', ' + str(self.batch_size) + ')' )
        n_validation_down = 0
        for n in xrange(self.n_epochs):
            for batch in iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                l_train, acc_train = self.f_train(x_batch, y_batch)
                l_val, acc_val = self.f_val(X_dev, Y_dev)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val) + ',best_val_acc ' + str(best_val_acc))
            if n_validation_down > 2:
                logging.info('validation results went down. early stopping ...')
                break
        
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        
        logging.info('***************** final results based on best validation **************')
        l_val, acc_val = self.f_val(X_dev, Y_dev)
        logging.info('Best dev acc: %f' %(acc_val))
        
    def predict(self, X_test):
        X_test = X_test.astype('float32')
        return self.f_predict(X_test)
    
    def predict_proba(self, X_test):
        X_test = X_test.astype('float32')
        return self.f_predict_proba(X_test)
    
    def accuracy(self, X_test, Y_test):
        X_test = X_test.astype('float32')
        if self.complete_prob:
            Y_test = Y_test.astype('float32')
        else:
            Y_test = Y_test.astype('int32')
        test_loss, test_acc = self.f_val(X_test, Y_test)
        return test_acc
    
def load_geolocation_data(mindf=10, complete_prob=True, regression=False):

    geolocate.initialize(granularity=params.BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=params.do_not_discretize)
    params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names = geolocate.feature_extractor(norm=params.norm, use_mention_dictionary=params.mention_only, min_df=mindf, max_df=0.1, stop_words='english', binary=True, sublinear_tf=False, vocab=None, use_idf=True, save_vectorizer=False, complete_prob=complete_prob)
    
    if regression:
        params.Y_train = np.array([geolocate.locationStr2Float(params.trainUsers[u]) for u in params.U_train])
        params.Y_test = np.array([geolocate.locationStr2Float(params.testUsers[u]) for u in params.U_test])
        params.Y_dev = np.array([geolocate.locationStr2Float(params.devUsers[u]) for u in params.U_dev])
        return params.X_train, params.Y_train, params.X_test, params.Y_test, params.X_dev, params.Y_dev
 
    return params.X_train, params.Y_train, params.X_test, params.Y_test, params.X_dev, params.Y_dev
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def nn_model(X_train, Y_train, X_test, Y_test, n_epochs=10, batch_size=1000, init_parameters=None, complete_prob=True, add_hidden=False, regul_coef=5e-5, save_results=True):
    
    logging.info('building the network...' + ' hidden:' + str(add_hidden))
    in_size = X_train.shape[1]
    
    if complete_prob:
        out_size = Y_train.shape[1]
    else:
        out_size = len(set(Y_train.tolist()))
    # Prepare Theano variables for inputs and targets
    if not sp.sparse.issparse(X_train):
        X_sym = T.matrix()
    else:
        X_sym = S.csr_matrix(name='inputs', dtype='float32')
    
    if complete_prob:
        y_sym = T.matrix()
    else:
        y_sym = T.ivector()    
    l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                     input_var=X_sym)
    
    drop_input = False
    if drop_input:
        l_in = lasagne.layers.dropout(l_in, p=0.2)

    if add_hidden:
        if not sp.sparse.issparse(X_train):
            l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=1000,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        else:
            l_hid1 = SparseInputDenseLayer(
                l_in, num_units=1000,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
            
        
        l_out = lasagne.layers.DenseLayer(
        l_hid1, num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)
    else:
        if not sp.sparse.issparse(X_train):
            l_out = lasagne.layers.DenseLayer(
                l_in, num_units=out_size,
                nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_out = SparseInputDenseLayer(
                l_in, num_units=out_size,
                nonlinearity=lasagne.nonlinearities.softmax)

    
    

    if add_hidden:
        embedding = lasagne.layers.get_output(l_hid1, X_sym)
        f_get_embeddings = theano.function([X_sym], embedding)
    output = lasagne.layers.get_output(l_out, X_sym)
    pred = output.argmax(-1)
    loss = lasagne.objectives.categorical_crossentropy(output, y_sym)
    #loss = lasagne.objectives.multiclass_hinge_loss(output, y_sym)
    
    l1_share = 0.9
    l1_penalty = lasagne.regularization.regularize_layer_params(l_out, l1) * regul_coef * l1_share
    l2_penalty = lasagne.regularization.regularize_layer_params(l_out, l2) * regul_coef * (1-l1_share)
    loss = loss + l1_penalty + l2_penalty
    loss = loss.mean()
    if complete_prob:
        y_sym_one_hot = y_sym.argmax(-1)
        acc = T.mean(T.eq(pred, y_sym_one_hot))
    else:
        acc = T.mean(T.eq(pred, y_sym))
    if init_parameters:
        lasagne.layers.set_all_param_values(l_out, init_parameters)
    parameters = lasagne.layers.get_all_params(l_out, trainable=True)
    
    #print(params)
    #updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.01, momentum=0.9)
    #updates = lasagne.updates.sgd(loss, parameters, learning_rate=0.01)
    #updates = lasagne.updates.adagrad(loss, parameters, learning_rate=0.1, epsilon=1e-6)
    #updates = lasagne.updates.adadelta(loss, parameters, learning_rate=0.1, rho=0.95, epsilon=1e-6)
    updates = lasagne.updates.adam(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
    f_val = theano.function([X_sym, y_sym], [loss, acc])
    f_predict = theano.function([X_sym], pred)
    f_predict_proba = theano.function([X_sym], output)
    
    
    do_scale = False
    #X_train = X_train.todense()
    #X_text = X_text.todense()
    if do_scale:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_text = scaler.transform(X_text)        
    #X = X_train.todense().astype(theano.config.floatX)
    #Xt = X_test.todense().astype(theano.config.floatX)
    X = X_train.astype('float32')
    Xt = X_test.astype('float32')
    #X = X_train.astype(theano.config.floatX)
    #Xt = X_test.astype(theano.config.floatX)
    if complete_prob:
        Y = Y_train.astype('float32')
        Yt = Y_test.astype('float32')
    else:
        Y = Y_train.astype('int32')
        Yt = Y_test.astype('int32')

    logging.info('training (n_epochs, batch_size) = (' + str(n_epochs) + ', ' + str(batch_size) + ')' )
    for n in xrange(n_epochs):
        for batch in iterate_minibatches(X, Y, batch_size, shuffle=True):
            x_batch, y_batch = batch
            l_train, acc_train = f_train(x_batch, y_batch)
        
        l_val, acc_val = f_val(Xt, Yt)
        logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val))
        geolocate.loss(f_predict(Xt), U_eval=params.U_dev, save_results=False )
    geolocate.loss(f_predict(Xt), U_eval=params.U_dev, save_results=save_results )
    logging.info( str(regul_coef))
    
    if add_hidden:
        #X_embs = f_get_embeddings(X)
        #Xt_embs = f_get_embeddings(Xt)
        pass
    train_probs = f_predict_proba(X)
    return train_probs
    #pdb.set_trace()

def nn_regression(X_train, Y_train, X_test, Y_test, n_epochs=10, batch_size=1000, init_parameters=None, complete_prob=True, add_hidden=False, regul_coef=5e-5, save_results=True):
    
    logging.info('building the network...' + ' hidden:' + str(add_hidden))
    in_size = X_train.shape[1]
    
    if complete_prob:
        out_size = Y_train.shape[1]
    else:
        out_size = len(set(Y_train.tolist()))
    # Prepare Theano variables for inputs and targets
    if not sp.sparse.issparse(X_train):
        X_sym = T.matrix()
    else:
        X_sym = S.csr_matrix(name='inputs', dtype='float32')
    
    if complete_prob:
        y_sym = T.matrix()
    else:
        y_sym = T.ivector()    
    l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                     input_var=X_sym)
    
    drop_input = False
    if drop_input:
        l_in = lasagne.layers.dropout(l_in, p=0.2)

    if add_hidden:
        layer2 = True
        if not sp.sparse.issparse(X_train):
            l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=1000,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        else:
            l_hid1 = SparseInputDenseLayer(
                l_in, num_units=1000,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
            if layer2:
                l_hid1 = lasagne.layers.DenseLayer(
                    l_hid1, num_units=300,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())
            
        
        l_out = lasagne.layers.DenseLayer(
        l_hid1, num_units=out_size,
        nonlinearity=None)
    else:
        if not sp.sparse.issparse(X_train):
            l_out = lasagne.layers.DenseLayer(
                l_in, num_units=out_size,
                nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_out = SparseInputDenseLayer(
                l_in, num_units=out_size,
                nonlinearity=lasagne.nonlinearities.softmax)

    
    

    if add_hidden:
        pass
        #embedding = lasagne.layers.get_output(l_hid1, X_sym)
        #f_get_embeddings = theano.function([X_sym], embedding)
    output = lasagne.layers.get_output(l_out, X_sym)
    #pred = output.argmax(-1)
    #loss = lasagne.objectives.categorical_crossentropy(output, y_sym)
    loss = lasagne.objectives.squared_error(output, y_sym)
    #loss = lasagne.objectives.multiclass_hinge_loss(output, y_sym)
    
    l1_share = 0.9
    l1_penalty = lasagne.regularization.regularize_layer_params(l_out, l1) * regul_coef * l1_share
    l2_penalty = lasagne.regularization.regularize_layer_params(l_out, l2) * regul_coef * (1-l1_share)
    loss = loss + l1_penalty + l2_penalty
    loss = loss.mean()

    if init_parameters:
        lasagne.layers.set_all_param_values(l_out, init_parameters)
    parameters = lasagne.layers.get_all_params(l_out, trainable=True)
    
    #print(params)
    #updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.01, momentum=0.9)
    #updates = lasagne.updates.sgd(loss, parameters, learning_rate=0.01)
    #updates = lasagne.updates.adagrad(loss, parameters, learning_rate=0.1, epsilon=1e-6)
    #updates = lasagne.updates.adadelta(loss, parameters, learning_rate=0.1, rho=0.95, epsilon=1e-6)
    updates = lasagne.updates.adam(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    f_train = theano.function([X_sym, y_sym], loss, updates=updates)
    f_val = theano.function([X_sym, y_sym], loss)
    f_predict_proba = theano.function([X_sym], output)

    X = X_train.astype('float32')
    Xt = X_test.astype('float32')
    #X = X_train.astype(theano.config.floatX)
    #Xt = X_test.astype(theano.config.floatX)
    if complete_prob:
        Y = Y_train.astype('float32')
        Yt = Y_test.astype('float32')


    logging.info('training (n_epochs, batch_size) = (' + str(n_epochs) + ', ' + str(batch_size) + ')' )
    for n in xrange(n_epochs):
        for batch in iterate_minibatches(X, Y, batch_size, shuffle=True):
            x_batch, y_batch = batch
            l_train = f_train(x_batch, y_batch)
        
        l_val = f_val(Xt, Yt)
        logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train)  + ' ,val_loss ' + str(l_val) )
        test_preds =  f_predict_proba(Xt)
        #logging.info(str(test_preds[0:10, :]))
        geolocate.loss_latlon(params.U_dev, test_preds)
    geolocate.loss_latlon(params.U_dev, f_predict_proba(Xt))
    logging.info( str(regul_coef))
    
    if add_hidden:
        #X_embs = f_get_embeddings(X)
        #Xt_embs = f_get_embeddings(Xt)
        pass
    train_probs = f_predict_proba(X)
    return train_probs
    #pdb.set_trace()
    
def get_dare_texts():
    texts = []
    vocabs = params.vectorizer.get_feature_names()
    texts = texts + vocabs
    #now read dare json files
    json_file = '/home/arahimi/datasets/dare/geodare.cleansed.filtered.json'
    json_objs = []
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            texts.append(word)
            if subregions:
                texts.append(subregions.lower())
                subregion_items = subregions.lower().split(',')
                subregion_items = [item.strip() for item in subregion_items]
                texts.extend(subregion_items)
            else:
                texts.append(dialect)
    texts = sorted(list(set([text.strip() for text in texts if len(text)>1])))
    return texts
def read_1m_words(input_file='/home/arahimi/datasets/1mwords/count_1w.txt'):
    wordcount = []
    with codecs.open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            word, count = line.strip().split('\t')
            wordcount.append((word, count))
    return wordcount

def nearest_neighbours(vocab, embs, k):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)

    #now read dare json files
    json_file = '/home/arahimi/datasets/dare/geodare.cleansed.filtered.json'
    json_objs = []
    texts = []
    dialect_words = defaultdict(list)
    dialect_subregions = {}
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            texts.append(word)
            dialect_words[dialect].append(word)
            if subregions:
                texts.append(subregions.lower())
                subregion_items = [item for item in subregions.lower().split(',') if len(item.strip()) > 0]
                dialect_subregions[dialect] = subregion_items
                texts.extend(subregion_items)
            else:
                texts.append(dialect)

    
    logging.info('creating dialect embeddings by multiplying subregion embdeddings')
    dialect_embs = OrderedDict()
    vocabset = set(vocab)
    for dialect in sorted(dialect_words):
        dialect_items = dialect_subregions.get(dialect, [dialect])
        extended_dialect_items = []
        for dialect_item in dialect_items:
            itemsplit = dialect_item.split()
            extended_dialect_items.extend(itemsplit)
        itemsplit = dialect.split()
        extended_dialect_items.extend(itemsplit)
        dialect_item_indices = [vocab.index(item) for item in extended_dialect_items if item in vocabset]
        dialect_emb = np.ones((1, embs.shape[1]))
        for _index in dialect_item_indices:
            dialect_emb *= embs[_index, :].reshape((1, embs.shape[1]))
        #dialect_emb = dialect_emb / len(dialect_item_indices)
        dialect_embs[dialect] = dialect_emb
    target_X = np.vstack(tuple(dialect_embs.values()))
    #logging.info('MinMax Scaling each dimension to fit between 0,1')
    #target_X = scaler.fit_transform(target_X)
    #logging.info('l1 normalizing embedding samples')
    #target_X = normalize(target_X, norm='l1', axis=1, copy=False)

    #target_indices = np.asarray(text_index.values())
    #target_X = embs[target_indices, :]
    logging.info('computing nearest neighbours of dialects')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=10).fit(embs)
    distances, indices = nbrs.kneighbors(target_X)
    word_nbrs = [(dialect_embs.keys()[i], vocab[indices[i, j]]) for i in range(target_X.shape[0]) for j in range(k)]
    word_neighbours = defaultdict(list)
    for word_nbr in word_nbrs:
        word, nbr = word_nbr
        word_neighbours[word].append(nbr)
    
    return word_neighbours

def calc_recall(word_nbrs, k, freqwords=set()):
    json_file = '/home/arahimi/datasets/dare/geodare.cleansed.filtered.json'
    json_objs = []
    texts = []
    dialect_words = defaultdict(list)
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            texts.append(word)
            dialect_words[dialect].append(word)

    recalls = []
    info = []
    total_true_positive = 0
    total_positive = 0
    for dialect, nbrs in word_nbrs.iteritems():
        dialect_has = 0
        dialect_total = 0
        nbrs = [nbr for nbr in nbrs if nbr not in freqwords and nbr[0]!='#']
        nbrs = set(nbrs[0:k])
        if dialect in dialect_words:
            dwords = set(dialect_words[dialect])
            dialect_total = len(dwords)
            total_positive += dialect_total
            if dialect_total == 0:
                print('zero dialect words ' + dialect)
                continue
            for dword in dwords:
                if dword in nbrs:
                    dialect_has += 1
                    total_true_positive += 1
            recall = 100 * float(dialect_has) / dialect_total
            recalls.append(recall)
            info.append((dialect, dialect_total, recall))
        else:
            print('this dialect does not exist: ' + dialect)
    print('recall at ' + str(k))
    #print(len(recalls))
    #print(np.mean(recalls))
    #print(np.median(recalls))
    #print(info)
    sum_support = sum([inf[1] for inf in info])
    #weighted_average_recall = sum([inf[1] * inf[2] for inf in info]) / sum_support
    #print('weighted average recall: ' + str(weighted_average_recall))
    print('micro recall :' + str(float(total_true_positive) * 100 / total_positive))

def geo_mlp(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, n_epochs=10, batch_size=1000, init_parameters=None, complete_prob=True, add_hidden=False, regul_coefs=[5e-5, 5e-5], save_results=True, hidden_layer_size=None, drop_out=False, drop_out_coefs=[0.5, 0.5]):
    
    logging.info('building the network...' + ' hidden:' + str(add_hidden))
    in_size = X_train.shape[1]
    best_params = None
    best_dev_acc = 0.0
    drop_out_hid, drop_out_in = drop_out_coefs
    if complete_prob:
        out_size = Y_train.shape[1]
    else:
        out_size = len(set(Y_train.tolist()))
    
    if hidden_layer_size:
        pass
    else:
        hidden_layer_size = min(5 * out_size, int(in_size / 20))
    logging.info('hidden layer size is ' + str(hidden_layer_size))
    # Prepare Theano variables for inputs and targets
    if not sp.sparse.issparse(X_train):
        logging.info('input matrix is not sparse!')
        X_sym = T.matrix()
    else:
        X_sym = S.csr_matrix(name='inputs', dtype='float32')
    
    if complete_prob:
        y_sym = T.matrix()
    else:
        y_sym = T.ivector()    
    
    l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                     input_var=X_sym)
    
    if drop_out:
        l_in = lasagne.layers.dropout(l_in, p=drop_out_in)

    if add_hidden:
        if not sp.sparse.issparse(X_train):
            l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=hidden_layer_size,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        else:
            l_hid1 = SparseInputDenseLayer(
                l_in, num_units=hidden_layer_size,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        if drop_out:
            l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
        
        l_out = lasagne.layers.DenseLayer(
        l_hid1, num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)
    else:
        if not sp.sparse.issparse(X_train):
            l_out = lasagne.layers.DenseLayer(
                l_in, num_units=out_size,
                nonlinearity=lasagne.nonlinearities.softmax)
            if drop_out:
                l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
        else:
            l_out = SparseInputDenseLayer(
                l_in, num_units=out_size,
                nonlinearity=lasagne.nonlinearities.softmax)
            if drop_out:
                l_hid1 = SparseInputDropoutLayer(l_hid1, drop_out_hid)
    
    

    if add_hidden:
        embedding = lasagne.layers.get_output(l_hid1, X_sym, deterministic=True)
        f_get_embeddings = theano.function([X_sym], embedding)
    output = lasagne.layers.get_output(l_out, X_sym, deterministic=True)
    pred = output.argmax(-1)
    loss = lasagne.objectives.categorical_crossentropy(output, y_sym)
    loss = loss.mean()
    #loss = lasagne.objectives.multiclass_hinge_loss(output, y_sym)
    
    l1_share_out = 0.5
    l1_share_hid = 0.5
    l1_penalty = 0.0
    l2_penalty = 0.0
    regul_coef_out, regul_coef_hid = regul_coefs
    logging.info('regul coefficient for output and hidden layers are ' + str(regul_coefs))
    l1_penalty = lasagne.regularization.regularize_layer_params(l_out, l1) * regul_coef_out * l1_share_out
    l2_penalty = lasagne.regularization.regularize_layer_params(l_out, l2) * regul_coef_out * (1-l1_share_out)
    if add_hidden:
        l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef_hid * l1_share_hid
        l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef_hid * (1-l1_share_hid)
    loss = loss + l1_penalty + l2_penalty

    if complete_prob:
        y_sym_one_hot = y_sym.argmax(-1)
        acc = T.mean(T.eq(pred, y_sym_one_hot))
    else:
        acc = T.mean(T.eq(pred, y_sym))
    if init_parameters:
        lasagne.layers.set_all_param_values(l_out, init_parameters)
    parameters = lasagne.layers.get_all_params(l_out, trainable=True)
    
    #print(params)
    #updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.01, momentum=0.9)
    #updates = lasagne.updates.sgd(loss, parameters, learning_rate=0.01)
    #updates = lasagne.updates.adagrad(loss, parameters, learning_rate=0.1, epsilon=1e-6)
    #updates = lasagne.updates.adadelta(loss, parameters, learning_rate=0.1, rho=0.95, epsilon=1e-6)
    updates = lasagne.updates.adam(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
    f_val = theano.function([X_sym, y_sym], [loss, acc])
    f_predict = theano.function([X_sym], pred)
    f_predict_proba = theano.function([X_sym], output)
    
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_dev = X_dev.astype('float32')

    if complete_prob:
        Y_train = Y_train.astype('float32')
        Y_test = Y_test.astype('float32')
        Y_dev = Y_dev.astype('float32')
    else:
        Y_train = Y_train.astype('int32')
        Y_test = Y_test.astype('int32')
        Y_dev = Y_dev.astype('int32')

    logging.info('training (n_epochs, batch_size) = (' + str(n_epochs) + ', ' + str(batch_size) + ')' )
    n_validation_down = 0
    for n in xrange(n_epochs):
        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
            x_batch, y_batch = batch
            l_train, acc_train = f_train(x_batch, y_batch)
            l_val, acc_val = f_val(X_dev, Y_dev)
        mean, median, acc_at_161 = geolocate.loss(f_predict(X_dev), U_eval=params.U_dev, save_results=False, verbose=False)
        if acc_at_161 > best_dev_acc:
            best_dev_acc = acc_at_161
            best_params = lasagne.layers.get_all_param_values(l_out)
            n_validation_down = 0
        else:
            #early stopping
            n_validation_down += 1
        l_val, acc_val = f_val(X_dev, Y_dev)
        logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val) + ',best_val_acc ' + str(best_dev_acc))
        logging.info('dev results after epoch')
        mean, median, acc_at_161 = geolocate.loss(f_predict(X_dev), U_eval=params.U_dev, save_results=False )
        logging.info('test results after epoch')
        mean, median, acc_at_161 = geolocate.loss(f_predict(X_test), U_eval=params.U_test, save_results=False )
        if n_validation_down > 2:
            logging.info('validation results went down. early stopping ...')
            break
    
    lasagne.layers.set_all_param_values(l_out, best_params)
    
    logging.info('***************** final results based on best validation **************')
    logging.info('dev results')
    devPreds = f_predict(X_dev)
    testPreds = f_predict(X_test)
    devProbs = f_predict_proba(X_dev)
    testProbs = f_predict_proba(X_test)
    mean, median, acc_at_161 = geolocate.loss(devPreds, U_eval=params.U_dev, save_results=False, error_analysis=False)
    logging.info('test results')
    mean, median, acc_at_161 = geolocate.loss(testPreds, U_eval=params.U_test, save_results=False )
    dump_preds = True
    if dump_preds:
        result_dump_file = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE) + '-mlp.pkl')
        print "dumping preds (preds, devPreds, params.U_test, params.U_dev, testProbs, devProbs) in " + result_dump_file
        with open(result_dump_file, 'wb') as outf:
            pickle.dump((testPreds, devPreds, params.U_test, params.U_dev, testProbs, devProbs), outf)

    if add_hidden:
        pass
        #X_train_embs = f_get_embeddings(X_train)
        #Xt_embs = f_get_embeddings(Xt)
        #logging.info('reading DARE vocab...')
        #dare_texts = get_dare_texts()
        #X_dare = params.vectorizer.transform(dare_texts)
        #X_dare = X_dare.astype('float32')
        #logging.info('getting DARE embeddings...')
        #X_dare_embs = f_get_embeddings(X_dare)
        #word_neighbours = nearest_neighbours(vocab=dare_texts, embs=X_dare_embs)
        #logging.info('writing word_neighbours dict...')
        #with open('./word_neighbour.pkl', 'wb') as fout:
        #    pickle.dump(word_neighbours, fout)
        #pdb.set_trace()
        #logging.info('writing dare_texts, X_dare_embs in word-embs.pkl')
        #logging.info('writing the embeddings...')
        #with open('/tmp/word-embs-' + str(batch_size) + '-' + str(regul_coef_out) + '-' + str(regul_coef_hid) + '.pkl', 'wb') as fout:
        #    pickle.dump((dare_texts, X_dare_embs), fout)
    train_probs = f_predict_proba(X_train)

    return train_probs

def load_word2vec(fname):
    import gensim
    ''' load a pre-trained binary format word2vec into a dictionary
    the model is downloaded from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download'''
    word2vec = gensim.models.word2vec.Word2Vec.load_word2vec_format(fname, binary=True)
    return word2vec


def dialect_eval(embs_file='./word-embs-10000-1e-06-1e-06.pkl.gz', word2vec=None, lr=None):
    logging.info('word2vec: ' + str(word2vec) + " lr: " + str(lr))
    logging.info('loading vocab, embs from ' + embs_file)
    with gzip.open(embs_file, 'rb') as fin:
        vocab, embs = pickle.load(fin)
    vocab_size = len(vocab)
    print('vocab size: ' + str(vocab_size))
    if word2vec:
        vocabset = set(vocab)
        logging.info('loading w2v embeddings...')
        word2vec_model = load_word2vec('/home/arahimi/GoogleNews-vectors-negative300.bin.gz')
        w2v_vocab = [v for v in word2vec_model.vocab if v in vocabset]
        logging.info('vstacking word vectors into a single matrix...')
        w2v_embs = np.vstack(tuple([np.asarray(word2vec_model[v]).reshape((1,300)) for v in w2v_vocab]))
        embs = w2v_embs
        vocab = w2v_vocab 
    elif lr:
        with open('/home/arahimi/datasets/na-original/model-na-original-median-2400-1e-06.pkl', 'rb') as fout:
            clf, vectorizer = pickle.load(fout)
        X_lr = vectorizer.transform(vocab)
        lr_embs = clf.predict_proba(X_lr) 
        embs = lr_embs       
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    logging.info('MinMax Scaling each dimension to fit between 0,1')
    #embs = scaler.fit_transform(embs)
    logging.info('l1 normalizing embedding samples')
    #embs = normalize(embs, norm='l1', axis=1, copy=False)
    word_nbrs = nearest_neighbours(vocab, embs, k=int(len(vocab)))
    wordfreqs = read_1m_words()
    topwords = [wordfreq[0] for wordfreq in wordfreqs]
    freqwords = set(topwords[0:50000])
    
    percents = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    percents = [int(p* vocab_size) for p in percents]
    for r_at_k in percents:
        calc_recall(word_nbrs=word_nbrs, k=r_at_k, freqwords=freqwords)

def retrieve_location_from_coordinates(points):
    """Given a list of coordinates, uses the geocoder and finds the corresponding
    locations and returns them in a dictionary. It requires internet connection
    to retrieve the addresses from openstreetmap.org. The geolocation service provider
    can be changed according to geopy documentation.
    
    Args:
        points (iterable): an iterable (e.g. a list) of tuples in the form of (lat, lon) where coordinates are of type float e.g. [(1.2343, -10.239834r),(5.634534, -12.47563)]
    Returns:
        co_loc (dict): a dictionary of the form point: location where location is itself a dictionary including the corresponding address of the point.
    
    """
    from geopy.geocoders import Nominatim
    geocoder = Nominatim(timeout=10)
    coordinate_location = {}
    
    for coordinate in points:
        try:
            location = geocoder.reverse(coordinate)
        except:
            location = 'unknown'
        coordinate_location[coordinate] = location
    co_loc = {k:v.raw for k,v in coordinate_location.iteritems()}
    
    return co_loc

def conf_matrix(y, pred):
    from sklearn.metrics import confusion_matrix
    labels = sorted(list(set(y + pred)))
    cm = confusion_matrix(y, pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion.pdf')
def error_analysis_text_correction():
    #from diagrams import draw_lines_on_basemap
    geolocate.initialize(granularity=params.BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=True)
    points = []
    rpoints = []
    ppoints = []
    with codecs.open('./www2017/error_analysis_geolocation - Sheet1.tsv', 'r', encoding='utf-8') as fin:
        for line in fin:
            fields = line.strip().split('\t')
            rpoint = (float(fields[2]), float(fields[3]))
            ppoint = (float(fields[4]), float(fields[5]))
            points.append(rpoint)
            rpoints.append(rpoint)
            ppoints.append(ppoint)
            points.append(ppoint)
    #with open('/home/arahimi/git/geolocation/www2017/points.pkl', 'wb') as fout:
    #    pickle.dump((rpoints, ppoints), fout)
    #draw_lines_on_basemap(rpoints[0:100], ppoints[0:100])
    #sys.exit()
    #co_loc = retrieve_location_from_coordinates(points)
    with open('/home/arahimi/git/geolocation/www2017/coordinate_address.pkl', 'rb') as fin:
        co_loc = pickle.load(fin)
    y = []
    pred = []
    with codecs.open('./www2017/error_analysis_geolocation - Sheet2.tsv', 'w', encoding='utf-8') as fout:
        with codecs.open('./www2017/error_analysis_geolocation - Sheet1.tsv', 'r', encoding='utf-8') as fin:
            for line in fin:
                fields = line.strip().split('\t')
                user = fields[0].lower()
                text = params.devText[user.lower()]
                rpoint = (float(fields[2]), float(fields[3]))
                ppoint = (float(fields[4]), float(fields[5]))
                rloc = co_loc[rpoint]
                ploc = co_loc[ppoint]
                rstate, rcity, pstate, pcity = ['', '', '', '']
                if 'address' in rloc:
                    rstate = rloc['address'].get('state',u'').encode('utf-8')
                    rcity = rloc['address'].get('city',u'').encode('utf-8')
                if 'address' in ploc:
                    pstate = ploc['address'].get('state',u'').encode('utf-8')
                    pcity = ploc['address'].get('city',u'').encode('utf-8')
                y.append(rstate)
                pred.append(pstate)
                newfields = fields[0:6] + [rstate , rcity, pstate , pcity, text]
                types = [type(f) for f in newfields]
                #print types
                newfields = [f.decode('utf-8') for f in newfields]
                newline = '\t'.join(newfields) + '\n'
                fout.write(newline)
    conf_matrix(y, pred)

if __name__ == '__main__':
    #error_analysis_text_correction()
    dialect_eval(word2vec=None, lr=True)
    sys.exit()
    regression = False
    do_tune = False
    add_hiden = True
    complete_prob = False
    X_train, Y_train, X_test, Y_test, X_dev, Y_dev = load_geolocation_data(mindf=10, complete_prob=complete_prob, regression=regression)
    n_epochs = 10
    if params.DATASET_NUMBER == 1:
        #first output then hidden layer coefficinet
        _coefs = [1e-5, 1e-5]
        batch_size = 100
        _hidden_layer_size = 32 * 28
    elif params.DATASET_NUMBER == 2:
        _coefs = [1e-6, 1e-6]
        batch_size = 10000
        _hidden_layer_size = 256 * 8
    else:
        _coefs = [1e-6, 1e-6]
        batch_size = 10000
        _hidden_layer_size = 930 * 4
    print('coefs: ' + str(_coefs) + ' batch_size: ' + str(batch_size))

    if regression:
        nn_regression(X_train, Y_train, X_test, Y_test, complete_prob=complete_prob, regul_coefs=_coefs, add_hidden=add_hiden, batch_size=batch_size,  n_epochs=n_epochs)
    else:
        geo_mlp(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, n_epochs=n_epochs, batch_size=batch_size, init_parameters=None, complete_prob=complete_prob, add_hidden=add_hiden, regul_coefs=_coefs, save_results=False, hidden_layer_size=_hidden_layer_size, drop_out=True, drop_out_coefs=[0.5, 0.5])            
            
    



