'''
Created on 10 May 2016

@author: af
'''
import numpy as np
from scipy.stats import multivariate_normal as mvn
import pdb
import matplotlib.pyplot as plt
from numpy.core.umath_tests import matrix_multiply as mm
from collections import defaultdict
from scipy.sparse import csr_matrix, vstack
import params
import geolocate

def get_text_model(X, y, n_classes):
    #pdb.set_trace()
    class_users = defaultdict(list)
    for i in range(X.shape[0]):
        class_users[y[i]].append(params.U_train[i])
    #find median lat and lon for each class
    for _class, _users in class_users.iteritems():
        locs = [geolocate.locationStr2Float(params.trainUsers[u]) for u in _users]
        medianLat = np.median([loc[0] for loc in locs])
        medianLon = np.median([loc[1] for loc in locs])
        params.classLatMedian[str(_class)] = medianLat
        params.classLonMedian[str(_class)] = medianLon 
        
    #some times some labels don't exist so we add these extra empty samples so that each label is occurred in training data at least once!
    extra_samples = csr_matrix((n_classes, X.shape[1]))
    params.X_train = vstack([params.X_train, extra_samples])
    params.Y_train = np.array(y.tolist() + list(range(n_classes)) )
    geolocate.classify(granularity=params.BUCKET_SIZE, DSExpansion=False, DSModification=False, compute_dev=True, report_verbose=False, clf=None, regul=params.reguls[params.DATASET_NUMBER-1], partitionMethod='median', penalty='elasticnet', fit_intercept=True, save_model=False, reload_model=False)
    #remove the extra samples and extra labels
    params.Y_train = params.Y_train[0:params.Y_train.shape[0]-n_classes]
    params.X_train = params.X_train[0:params.X_train.shape[0]-n_classes, :]
    print params.Y_train.shape
    print params.X_train.shape
    return params.clf.predict_proba(X)

def em_gmm_vect(points, cluster_weights, cluster_mu, cluster_sigmas, tol=0.01, max_iter=10, W=None, P_c_given_w=None):
    
    n_samples, n_dimensions = points.shape
    n_classes = len(cluster_weights)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((n_classes, n_samples))
        for j in range(n_classes):
            ws[j, :] = cluster_weights[j] * mvn(cluster_mu[j], cluster_sigmas[j], allow_singular=True).pdf(points)
        ws /= ws.sum(0)
        
        if W is not None:
            multiply = True
            if multiply:
                ws = np.multiply(ws, np.transpose(P_c_given_w))
            else:
                #add (can be weighted)
                ws_weight = 0.8
                ws = ws_weight * ws + (1-ws_weight) * np.transpose(P_c_given_w)
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
        ll_new = 0
        for pi, mu, sigma in zip(cluster_weights, cluster_mu, cluster_sigmas):
            ll_new += pi*mvn(mu, sigma, allow_singular=True).pdf(points)
        ll_new = np.log(ll_new + 1e-5).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new
        print 'iter', i,'change in ll', np.abs(ll_new - ll_old)
        if W is not None:
            y = np.argmax(ws, axis=0)
            P_c_given_w = get_text_model(W, y, n_classes)

    return ll_new, cluster_weights, cluster_mu, cluster_sigmas

geolocate.initialize(granularity=params.BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=params.do_not_discretize)
params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names = geolocate.feature_extractor(norm=params.norm, use_mention_dictionary=False, min_df=10, max_df=0.2, stop_words='english', binary=True, sublinear_tf=False, vocab=None, use_idf=True, save_vectorizer=False)
geolocate.classify(granularity=params.BUCKET_SIZE, DSExpansion=False, DSModification=False, compute_dev=True, report_verbose=False, clf=None, regul=params.reguls[params.DATASET_NUMBER-1], partitionMethod='median', penalty='elasticnet', fit_intercept=True, save_model=False, reload_model=False)
   
'''    
np.random.seed(123)

# create data set
n_samples = 1000
_mus = np.array([[0,4], [-2,0]])
_sigmas = np.array([[[3, 0], [0, 0.5]], [[1,0],[0,2]]])
_pis = np.array([0.6, 0.4])
xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*n_samples))
                    for pi, mu, sigma in zip(_pis, _mus, _sigmas)])

# initial guesses for parameters
pis = np.random.random(2)
pis /= pis.sum()
mus = np.random.random((2,2))
sigmas = np.array([np.eye(2)] * 2)
'''
####################
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
ll2, pis2, mus2, sigmas2 = em_gmm_vect(xs, pis, mus, sigmas,W=params.X_train, P_c_given_w=params.clf.predict_proba(params.X_train))

'''
intervals = 101
ys = np.linspace(-8,8,intervals)
X, Y = np.meshgrid(ys, ys)
_ys = np.vstack([X.ravel(), Y.ravel()]).T

z = np.zeros(len(_ys))
for pi, mu, sigma in zip(pis2, mus2, sigmas2):
    z += pi*mvn(mu, sigma).pdf(_ys)
z = z.reshape((intervals, intervals))

ax = plt.subplot(111)
plt.scatter(xs[:,0], xs[:,1], alpha=0.2)
plt.contour(X, Y, z, N=10)
plt.axis([-8,6,-6,8])
ax.axes.set_aspect('equal')
plt.tight_layout()
plt.show()
'''