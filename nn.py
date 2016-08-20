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


def load_data():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    #categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    categories = None 
    #remove = ('headers', 'footers', 'quotes')
    remove = ()
    train_set = fetch_20newsgroups(subset='train', 
                                   remove=remove,
                                   categories=categories)
    test_set = fetch_20newsgroups(subset='test', 
                                  remove=remove,
                                  categories=categories)
    Y_train, Y_test = train_set.target, test_set.target
    vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
    X_train = vectorizer.fit_transform(train_set.data)
    X_test = vectorizer.transform(test_set.data)
    return X_train, Y_train, X_test, Y_test

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
def geo_mlp(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, n_epochs=10, batch_size=1000, init_parameters=None, complete_prob=True, add_hidden=False, regul_coef=5e-5, save_results=True, hidden_layer_size=None):
    
    logging.info('building the network...' + ' hidden:' + str(add_hidden))
    in_size = X_train.shape[1]
    best_params = None
    best_dev_acc = 0.0
    
    if complete_prob:
        out_size = Y_train.shape[1]
    else:
        out_size = len(set(Y_train.tolist()))
    
    if hidden_layer_size:
        pass
    else:
        hidden_layer_size = 4 * out_size
    logging.info('hidden layer size is ' + str(hidden_layer_size))
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
                l_in, num_units=hidden_layer_size,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        else:
            l_hid1 = SparseInputDenseLayer(
                l_in, num_units=hidden_layer_size,
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
    if add_hidden:
        l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef * l1_share
        l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef * (1-l1_share)
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
        else:
            #early stopping
            n_validation_down += 1
            if n_validation_down > 1:
                logging.info('validation results went down. early stopping ...')
                break
        l_val, acc_val = f_val(X_dev, Y_dev)
        logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val) + ',best_val_acc ' + str(best_dev_acc))
        logging.info('dev results after epoch')
        mean, median, acc_at_161 = geolocate.loss(f_predict(X_dev), U_eval=params.U_dev, save_results=False )
        logging.info('test results after epoch')
        mean, median, acc_at_161 = geolocate.loss(f_predict(X_test), U_eval=params.U_test, save_results=False )
    
    lasagne.layers.set_all_param_values(l_out, best_params)
    
    logging.info('***************** final results based on best validation **************')
    logging.info('dev results')
    mean, median, acc_at_161 = geolocate.loss(f_predict(X_dev), U_eval=params.U_dev, save_results=False )
    logging.info('test results')
    mean, median, acc_at_161 = geolocate.loss(f_predict(X_test), U_eval=params.U_test, save_results=False )
    
    if add_hidden:
        #X_embs = f_get_embeddings(X)
        #Xt_embs = f_get_embeddings(Xt)
        pass
    train_probs = f_predict_proba(X_train)
    pdb.set_trace()
    return train_probs
if __name__ == '__main__':
    #X_train, Y_train, X_test, Y_test = load_data()
    regression = False
    do_tune = False
    add_hiden = True
    complete_prob = False
    X_train, Y_train, X_test, Y_test, X_dev, Y_dev = load_geolocation_data(complete_prob=complete_prob, regression=regression)
    n_epochs = 10
    
    if params.DATASET_NUMBER == 1:
        _coef = 1e-4
        batch_size = 10
    elif params.DATASET_NUMBER == 2:
        _coef = 1e-6
        batch_size = 1000
    else:
        _coef= 1e-6
        batch_size = 10000
    
    def tune(_coefs):
        for _coef in _coefs:
        #for _coef in [1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4]:
        #for _coef in [_coef]: 
            logging.info( str(_coef))
            if regression:
                nn_regression(X_train, Y_train, X_test, Y_test, complete_prob=complete_prob, regul_coef=_coef, add_hidden=add_hiden, batch_size=batch_size,  n_epochs=n_epochs)
            else:
                nn_model(X_train, Y_train, X_test, Y_test, complete_prob=complete_prob, regul_coef=_coef, add_hidden=add_hiden, batch_size=batch_size,  n_epochs=n_epochs)
    if do_tune:
        tune(_coefs=[10 ** -p for  p in range(4,8)])
    else:
        if regression:
            nn_regression(X_train, Y_train, X_test, Y_test, complete_prob=complete_prob, regul_coef=_coef, add_hidden=add_hiden, batch_size=batch_size,  n_epochs=n_epochs)
        else:
            #nn_model(X_train, Y_train, X_test, Y_test, complete_prob=complete_prob, regul_coef=_coef, add_hidden=add_hiden, batch_size=batch_size,  n_epochs=n_epochs)
            geo_mlp(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, n_epochs=n_epochs, batch_size=batch_size, init_parameters=None, complete_prob=complete_prob, add_hidden=add_hiden, regul_coef=_coef, save_results=False)
    ####pdb.set_trace()
    



