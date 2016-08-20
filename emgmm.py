'''
Created on 10 May 2016

@author: af
'''
import numpy as np
from scipy.stats import multivariate_normal as mvn
import pdb
from numpy.core.umath_tests import matrix_multiply as mm
from collections import defaultdict
from scipy.sparse import csr_matrix, vstack
import params
import geolocate
from os import path
import nn
import pickle
from diagrams import draw_points, line_chart, rand_cmap
from datetime import datetime
draw_maps = False
hard_em = False
regul_coef = 1e-6
batch_size = 1000
n_epochs = 2
add_hidden = True
max_iter = 20
topk = 0
print 'regul:', regul_coef, 'batch', batch_size, 'epochs', n_epochs, 'has hidden', add_hidden, 'topk', topk, 'max_iter', max_iter

def get_text_model_hard(X, y, n_classes, iter_number):
    #pdb.set_trace()
    print '#num of assigned classes: ', len(set(y.tolist()))
    class_users = defaultdict(list)
    for i in range(X.shape[0]):
        class_users[y[i]].append(params.U_train[i])
    #find median lat and lon for each class
    class_latlon = {}
    for _class, _users in class_users.iteritems():
        locs = [geolocate.locationStr2Float(params.trainUsers[u]) for u in _users]
        class_latlon[_class] = locs
        medianLat = np.median([loc[0] for loc in locs])
        medianLon = np.median([loc[1] for loc in locs])
        params.classLatMedian[str(_class)] = medianLat
        params.classLonMedian[str(_class)] = medianLon 
    if draw_maps:
        draw_points(class_latlon, filename='./maps/' + str(iter_number) + '.jpg')   
    #some times some labels don't exist so we add these extra empty samples so that each label is occurred in training data at least once!
    extra_samples = csr_matrix((n_classes, X.shape[1]))
    params.X_train = vstack([params.X_train, extra_samples])
    params.Y_train = np.array(y.tolist() + list(range(n_classes)) )
    geolocate.classify(granularity=params.BUCKET_SIZE, DSExpansion=False, DSModification=False, compute_dev=True, report_verbose=False, clf=None, regul=params.reguls[params.DATASET_NUMBER-1], partitionMethod='median', penalty='elasticnet', fit_intercept=True, save_model=False, reload_model=False)
    #remove the extra samples and extra labels
    params.Y_train = params.Y_train[0:params.Y_train.shape[0]-n_classes]
    params.X_train = params.X_train[0:params.X_train.shape[0]-n_classes, :]

    return params.clf.predict_proba(X)

def get_text_model_soft(iter_number):
    #pdb.set_trace()
    n_classes = params.Y_train.shape[1]
    y = np.argmax(params.Y_train, axis=1)
    n_assigned_classes = len(set(y.tolist()))
    print '#num of assigned classes: ', n_assigned_classes
    do_extra_assignment = False
    if n_assigned_classes < params.Y_train.shape[1]:
        do_extra_assignment = True
        
    print np.sum(params.Y_train, axis=0)
    class_users = defaultdict(list)
    for i in range(params.X_train.shape[0]):
        class_users[y[i]].append(params.U_train[i])
    #find median lat and lon for each class
    class_latlon = {}
    for _class, _users in class_users.iteritems():
        locs = [geolocate.locationStr2Float(params.trainUsers[u]) for u in _users]
        class_latlon[_class] = locs
        medianLat = np.median([loc[0] for loc in locs])
        medianLon = np.median([loc[1] for loc in locs])
        params.classLatMedian[str(_class)] = medianLat
        params.classLonMedian[str(_class)] = medianLon 
    if draw_maps:
        draw_points(class_latlon, filename='./maps/' + str(iter_number) + '.jpg', cm=random_cmap_new)   
    
    if do_extra_assignment:
        #some times some labels don't exist so we add these extra empty samples so that each label is occurred in training data at least once!
        extra_samples = csr_matrix((n_classes, params.X_train.shape[1]))
        extra_targets = np.zeros((n_classes, n_classes), float)
        np.fill_diagonal(extra_targets, 1)
        params.X_train = vstack([params.X_train, extra_samples])
        params.Y_train = np.vstack((params.Y_train, extra_targets))
    
    #train again using the provided labels and return new labels
    params.Y_train = nn.nn_model(params.X_train, params.Y_train, params.X_dev, params.Y_dev, n_epochs=n_epochs, batch_size=batch_size, init_parameters=None, complete_prob=True, add_hidden=add_hidden, regul_coef=regul_coef)
    if do_extra_assignment:
        #remove the extra samples and extra labels
        params.Y_train = params.Y_train[0:params.Y_train.shape[0]-n_classes, :]
        params.X_train = params.X_train[0:params.X_train.shape[0]-n_classes, :]
    print params.X_train.shape, params.Y_train.shape
    return params.Y_train 

def em_gmm_vect(points, cluster_weights, cluster_mu, cluster_sigmas, tol=10, max_iter=max_iter, W=None, P_c_given_w=None):
    
    n_samples, n_dimensions = points.shape
    n_classes = len(cluster_weights)
    lls = []
    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((n_classes, n_samples))
        no_cluster_weight = True
        for j in range(n_classes):
            if no_cluster_weight:
                ws[j, :] = mvn(cluster_mu[j], cluster_sigmas[j], allow_singular=True).pdf(points)
            else:
                ws[j, :] = cluster_weights[j] * mvn(cluster_mu[j], cluster_sigmas[j], allow_singular=True).pdf(points)
        ws /= ws.sum(0)
        
        if W is not None:
            ws = np.multiply(ws, np.transpose(P_c_given_w))
            ws /= ws.sum(0)
        

            

        # M-step
        cluster_weights = ws.sum(axis=1)
        cluster_weights /= n_samples

        cluster_mu = np.dot(ws, points)
        cluster_mu /= ws.sum(1)[:, None]

        cluster_sigmas = np.zeros((n_classes, n_dimensions, n_dimensions))
        for j in range(n_classes):
            ys = points - cluster_mu[j, :]
            cluster_sigmas[j] = (ws[j,:,None,None] * mm(ys[:,:,None], ys[:,None,:])).sum(axis=0)
        cluster_sigmas /= ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        pi_or_p_c_given_w = 'p_c_given_w'
        ll_new = 0
        _cluster_index = 0
        for pi, mu, sigma in zip(cluster_weights, cluster_mu, cluster_sigmas): 
            if pi_or_p_c_given_w == 'pi':
                ll_this_iter = pi*mvn(mu, sigma, allow_singular=True).pdf(points)
            else:
                ll_this_iter = np.multiply(P_c_given_w[:, _cluster_index] , mvn(mu, sigma, allow_singular=True).pdf(points))
            _cluster_index += 1

            #ll_this_iter[ll_this_iter == 0] = 1e-5
            ll_new += ll_this_iter
               
        ll_new = np.log(ll_new).sum()
        
        if np.abs(ll_new - ll_old) < tol:
            break
        
        print 'iter', i,'change in ll', int(np.abs(ll_new))
        ll_old = ll_new
        lls.append(np.abs(ll_new))
        if W is not None:
            if hard_em:
                y = np.argmax(ws, axis=0)
                P_c_given_w = get_text_model_hard(W, y, n_classes, iter_number=i+1)
            elif topk > 0:
                topk_indices = np.fliplr(np.argsort(np.transpose(ws), axis=1))[:, 0:topk]
                params.Y_train[:, :] = 0
                for _index in range(params.Y_train.shape[0]):
                    for _k in range(topk):
                        column = topk_indices[_index, _k]
                        params.Y_train[_index, column] = ws[column, _index]
                params.Y_train /= params.Y_train.sum(1).reshape(params.Y_train.shape[0], 1)
                get_text_model_soft(iter_number=i+1)
                
                
            else:
                params.Y_train = np.transpose(ws)
                get_text_model_soft(iter_number=i+1)
    #i += 1
    #params.Y_train = np.transpose(ws)
    #get_text_model_soft(iter_number=i+1)
    with open('./emgmm-model/' + str(add_hidden) + '-' + str(regul_coef)+ '-' + str(i) + '-' + datetime.now().strftime("lbl_dist.%Y-%m-%d-%H.pkl"), 'wb') as outf:
        pickle.dump((params.Y_train, ws, params.trainUsers, cluster_mu, cluster_sigmas, cluster_weights), outf)
    line_chart(means=params.dev_mean, medians=params.dev_median, accs=params.dev_acc, lls=lls, iters=list(range(len(lls)+1)), filename='./maps/' + str(add_hidden) + '-' + str(regul_coef)+ '-' + str(i) + '-' + datetime.now().strftime("lbl_dist.%Y-%m-%d-%H.pkl") + 'monitor.jpg')
    
    return ll_new, cluster_weights, cluster_mu, cluster_sigmas

if __name__ == '__main__':
    random_cmap_new = None
    if hard_em:
        geolocate.initialize(granularity=params.BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=params.do_not_discretize)
        params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names = geolocate.feature_extractor(norm=params.norm, use_mention_dictionary=False, min_df=10, max_df=0.2, stop_words='english', binary=True, sublinear_tf=False, vocab=None, use_idf=True, save_vectorizer=False)
        geolocate.classify(granularity=params.BUCKET_SIZE, DSExpansion=False, DSModification=False, compute_dev=True, report_verbose=False, clf=None, regul=params.reguls[params.DATASET_NUMBER-1], partitionMethod='median', penalty='elasticnet', fit_intercept=True, save_model=False, reload_model=False)
        class_locs = defaultdict(list)
        for _u, _c in params.trainClasses.iteritems():
            loc = geolocate.locationStr2Float(params.trainUsers[_u])
            class_locs[_c].append(loc)
        draw_points(class_locs, filename='./maps/0.jpg')
    else:
        X_train, Y_train, X_dev, Y_dev = nn.load_geolocation_data(complete_prob=True)
        random_cmap_new = rand_cmap(Y_train.shape[1])
        params.Y_train = Y_train
        params.Y_dev = Y_dev
        if draw_maps:
            class_locs = defaultdict(list)
            for _u, _c in params.trainClasses.iteritems():
                loc = geolocate.locationStr2Float(params.trainUsers[_u])
                class_locs[_c].append(loc)
                draw_points(class_locs, filename='./maps/0.jpg', cm=random_cmap_new)
        p_c_given_w = nn.nn_model(params.X_train, params.Y_train, params.X_dev, params.Y_dev, n_epochs=n_epochs, batch_size=batch_size, init_parameters=None, complete_prob=True, add_hidden=add_hidden, regul_coef=regul_coef)
        params.Y_train = p_c_given_w
        
       
    
    xs = np.array([geolocate.locationStr2Float(params.trainUsers[u]) for u in params.U_train] )
    num_classes = len(params.categories)
    pis = np.ones(num_classes)
    pis /= pis.sum()
    mus = np.array([[params.classLatMedian[str(i)], params.classLonMedian[str(i)]] for i in range(num_classes)])
    class_users = defaultdict(list)
    for u, _class in params.trainClasses.iteritems():
        lat, lon = geolocate.locationStr2Float(params.trainUsers[u])
        class_users[_class].append([lat, lon])
    sigmas = np.array([np.zeros((2,2))] * num_classes)
    for _class in range(num_classes):
        users = np.array(class_users[_class])
        sigma = np.cov(m=users, rowvar=0)
        sigmas[_class, :, :] = sigma
    
    
    
    #pdb.set_trace()
    if hard_em:
        ll2, pis2, mus2, sigmas2 = em_gmm_vect(xs, pis, mus, sigmas,W=params.X_train, P_c_given_w=params.clf.predict_proba(params.X_train))
    else:
        ll2, pis2, mus2, sigmas2 = em_gmm_vect(xs, pis, mus, sigmas,W=params.X_train, P_c_given_w=p_c_given_w)
    
