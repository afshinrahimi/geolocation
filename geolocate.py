'''
Created on 26 Feb 2016

@author: af
'''
'''
Created on 4 Sep 2014

@author: af
'''
from GPy import kern, likelihoods
import GPy
from GPy.core.gp import GP
from GPy.examples import regression
from IPython.core.debugger import Tracer
from _collections import defaultdict
import cPickle
import codecs
import codecs
from collections import Counter
from colorama import init, Fore, Back, Style
import copy
import csv
from datetime import datetime
from datetime import datetime
import glob
import gzip
import gzip
from haversine import haversine
import itertools
import logging
from math import radians, cos, sin, asin, sqrt
from math import radians, sin, cos, sqrt, asin
from math import sqrt
import math
from matplotlib.ticker import NullFormatter
import nltk
from numpy import float16, float32, float64
import numpy
from os import path
import os
import os
import pdb
import pickle
import random
import re
from scipy import linalg
from scipy import mean
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy.sparse.lil import lil_matrix
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import pdist
from scipy.stats import logistic
from scipy.stats import threshold
import sets
import shutil
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics
from sklearn import mixture, ensemble
from  sklearn.datasets import load_files
from sklearn.datasets import dump_svmlight_file
from sklearn.decomposition import *
from sklearn.decomposition import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, strip_accents_ascii, strip_accents_unicode
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.coordinate_descent import MultiTaskLasso, ElasticNet
from sklearn.linear_model.sgd_fast import Log
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.semi_supervised import *
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.extmath import density
import string
import sys
import sys
import theano
from theano.tensor.basic import dmatrix
import time

import matplotlib as mpl
import  matplotlib.collections as collections
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import numpy as np
from params import *
import pylab as pb
import scipy.sparse as sparse
import theano.tensor as T
import weighted as wst
import weightedstats as ws


mpl.use('Agg')
# from sklearn.linear_model import LogisticRegression
# this is just for printing colored in shell, you don't want it you comment it and its init line
init()

# from extract import get_tokens
# from time import time
pb.ion()
# from GPy.models_modules.gp_regression import GPRegression
__docformat__ = 'restructedtext en'
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# rootLogger = logging.getLogger()

# fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'log.log'))
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)

# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

print str(datetime.now())
script_start_time = time.time()


            
            
    
def median(mylist):
    return np.median(mylist)
    '''
    my implementation of median
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]
    '''
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    '''
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    '''
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    
    return haversine(point1, point2)
    # downloaded from http://rosettacode.org/wiki/Haversine_formula#Python
    R = 6372.8  # Earth radius in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))
    
    return R * c
    # return km 

def users(file, type='train', write=False, readText=True):
    if readText:
        print("Text is being read.")
        if TEXT_ONLY:
            print('mentions are removed.')

    # with codecs.open(file, 'r', encoding=data_encoding) as inf:
    with gzip.open(file, 'r') as inf:
        for line in inf:
            # print line
            fields = line.split('\t')
            if len(fields) != 4:
                print fields
            user = fields[0].strip().lower()
            lat = fields[1]
            lon = fields[2]
            if readText:
                text = fields[3].strip()
            if TEXT_ONLY and readText:
                text = re.sub(r"(?:\@|https?\://)\S+", "", text)
            locStr = lat + ',' + lon
            locFloat = (float(lat), float(lon))
            userLocation[user] = locStr
            if type == 'train':
                trainUsers[user] = locStr
                if readText:
                    trainText[user] = text
                users_in_loc = locationUser.get(locFloat, [])
                users_in_loc.append(user)
                locationUser[locFloat] = users_in_loc
            elif type == 'test':
                testUsers[user] = locStr
                if readText:
                    testText[user] = text
            elif type == 'dev':
                devUsers[user] = locStr
                if readText:
                    devText[user] = text

        








def createTrainDir(granularity, partitionMethod, create_dir=False):
    filename = path.join(GEOTEXT_HOME, 'processed_data/' + str(granularity).strip() + '_' + partitionMethod + '_clustered.train')
    print "reading " + filename
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    with codecs.open(filename, 'r', encoding=data_encoding) as inf:
        for line in inf:
            points = []
            minlat = 1000
            maxlat = -1000
            minlon = 1000
            maxlon = -1000
            fields = line.split('\t')
            points = [locationStr2Float(loc) for loc in fields]
            
            lats = [point[0] for point in points]
            lons = [point[1] for point in points]
            minlat = min(lats)
            maxlat = max(lats)
            minlon = min(lons)
            maxlon = max(lons)
            allpointsMinLat.append(minlat)
            allpointsMaxLat.append(maxlat)
            allpointsMaxLon.append(maxlon)
            allpointsMinLon.append(minlon)
            allpoints.append(points)

    i = -1
    for cluster in allpoints:
        # create a directory
        i += 1
        lats = [location[0] for location in cluster]
        longs = [location[1] for location in cluster]
        medianlat = np.median(lats)
        medianlon = np.median(longs)
        meanlat = np.mean(lats)
        meanlon = np.mean(longs)
        label = str(i).strip()
        categories.append(label)
        classLatMedian[label] = medianlat
        classLonMedian[label] = medianlon
        classLatMean[label] = meanlat
        classLonMean[label] = meanlon    
        
        for location in cluster:
            locusers = locationUser[(location[0], location[1])]
            user_class = dict(zip(locusers, [i] * len(locusers)))
            trainClasses.update(user_class)
    print "train directories created and class median and mean lat,lon computed. trainfile: " + filename

    devDistances = []
    for user in devUsers:
        locationStr = devUsers[user]
        latlon = locationStr.split(',')
        latitude = float(latlon[0])
        longitude = float(latlon[1])
        classIndex, dist = assignClass(latitude, longitude)
        devDistances.append(dist)
        devClasses[user] = int(classIndex)
    
    testDistances = []
    for user in testUsers:
        locationStr = testUsers[user]
        latlon = locationStr.split(',')
        latitude = float(latlon[0])
        longitude = float(latlon[1])
        classIndex, dist = assignClass(latitude, longitude)
        testDistances.append(dist)
        testClasses[user] = int(classIndex)

            


        
    Report_Ideal = False
    if Report_Ideal:
        print "Ideal mean dev distance is " + str(np.mean(devDistances))
        print "Ideal median dev distance is " + str(np.median(devDistances))
        print "Ideal Acc@161 dev is " + str(len([dist for dist in devDistances if dist < 161]) / (len(devDistances) + 0.0))
        
        print "Ideal mean test distance is " + str(np.mean(testDistances))
        print "Ideal median test distance is " + str(np.median(testDistances))
        print "Ideal Acc@161 test is " + str(len([dist for dist in testDistances if dist < 161]) / (len(testDistances) + 0.0))

def assignClass(latitude, longitude):
    '''
    Given a coordinate find the class whose median is the closest point. Then return the index of that class.
    This function can be used for parameter tuning with validation data and evaluation with test data.
    '''
    minDistance = 1000000
    classIndex = -1
    for i in classLatMedian:
        lat = classLatMedian[str(i).strip()]
        lon = classLonMedian[str(i).strip()]
        dist = distance(latitude, longitude, lat, lon)
        if dist < minDistance:
            minDistance = dist
            classIndex = i
    return classIndex, minDistance



def create_directories(granularity, partitionMethod, write=False):
    createTrainDir(granularity, partitionMethod, write)

def size_mb(docs):
    return sum(len(s.encode(encoding=data_encoding)) for s in docs) / 1e6

def evaluate(preds, U_test, categories, scores):
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    confidences = []
    randomConfidences = []
    gmm = False
    for i in range(0, len(preds)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        # gaussian mixture model
        if gmm:
            sumMedianLat = 0
            sumMedianLon = 0
            sumMeanLat = 0
            sumMeanLon = 0
            numClasses = len(categories)
            sortedScores = sorted(scores[i], reverse=True)
            top1Score = sortedScores[0]
            top2Score = sortedScores[1]
            print top1Score
            print top2Score
            for c in range(0, numClasses):
                score = scores[i][c]
                category = categories[c]
                medianlat = classLatMedian[category]  
                medianlon = classLonMedian[category]  
                meanlat = classLatMean[category] 
                meanlon = classLonMean[category]
                sumMedianLat += score * medianlat
                sumMedianLon += score * medianlon
                sumMeanLat += score * meanlat
                sumMeanLon += score * meanlon
            distances.append(distance(lat, lon, sumMedianLat, sumMedianLon)) 
            
        else:
            prediction = categories[preds[i]]
            if scores != None:
                confidence = scores[i][preds[i]] 
                confidences.append(confidence)
            medianlat = classLatMedian[prediction]  
            medianlon = classLonMedian[prediction]  
            meanlat = classLatMean[prediction] 
            meanlon = classLonMean[prediction]      
            distances.append(distance(lat, lon, medianlat, medianlon))
            sumMedianDistance = sumMedianDistance + distance(lat, lon, medianlat, medianlon)
            sumMeanDistance = sumMeanDistance + distance(lat, lon, meanlat, meanlon)
    # averageMeanDistance = sumMeanDistance / float(len(preds))
    # averageMedianDistance = sumMedianDistance / float(len(preds))
    # print "Average mean distance is " + str(averageMeanDistance)
    # print "Average median distance is " + str(averageMedianDistance)
    print "Mean distance is " + str(np.mean(distances))
    print "Median distance is " + str(np.median(distances))



            
def error(predicted_label, user):
    lat1, lon1 = locationStr2Float(userLocation[user])
    lat2 = classLatMedian[predicted_label]  
    lon2 = classLonMedian[predicted_label]
    return distance(lat1, lon1, lat2, lon2)         

def loss(preds, U_test, loss='median', save=False, save_results_bucket_size=False, results_key=None):
    global results
    if len(preds) != len(U_test): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_test))
        print "fatal error!"
        sys.exit()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    user_location = {}
    acc = 0.0
    center_of_us = (39.50, -98.35)
    nyc = (40.7127, -74.0059)
    la = (34.0500, -118.2500)
    distances_from_nyc = []
    distances_from_la = []
    distances_from_center = []
    heatmap_lats = []
    heatmap_lons = []
    heatmap_values = []
    uniformer = defaultdict(int)
    max_user_per_region = 50
    class_error = defaultdict(list)
    error_heatmap = False
    for i in range(0, len(preds)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        user_original_class, minDistance = assignClass(lat, lon)
        distances_from_center.append(distance(lat, lon, center_of_us[0], center_of_us[1]))
        distances_from_nyc.append(distance(lat, lon, nyc[0], nyc[1]))
        distances_from_la.append(distance(lat, lon, la[0], la[1]))
        # if preds[i] == int(testClasses[user]):
        #    acc += 1
        # print str(Y_test[i]) + " " + str(preds[i])
        prediction = categories[preds[i]]
        medianlat = classLatMedian[prediction]  
        medianlon = classLonMedian[prediction]  
        user_location[user] = (medianlat, medianlon)
        meanlat = classLatMean[prediction] 
        meanlon = classLonMean[prediction]
        predictionCoordinate = 'median'
        if predictionCoordinate == 'median':      
            dd = distance(lat, lon, medianlat, medianlon)
            distances.append(dd)
            if error_heatmap:
                num_in_region = uniformer[str(int(lat)) + ',' + str(int(lon))]
                heatmap_lats.append(lat)
                heatmap_lons.append(lon)
                heatmap_values.append(dd)
                uniformer[str(int(lat)) + ',' + str(int(lon))] = num_in_region + 1
                class_error[user_original_class].append(dd)
            
        elif predictionCoordinate == 'mean':
            distances.append(distance(lat, lon, meanlat, meanlon))
    
    if save:
        print "dumping the results in preds.pkl"
        with open(path.join(GEOTEXT_HOME, 'preds.pkl'), 'wb') as outf:
            pickle.dump(user_location, outf) 

    # print "Average distance from class mean is " + str(averageMeanDistance)
    # print "Average distance from class median is " + str(averageMedianDistance)
    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    print "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161))
    print
    extra_info = False
    if extra_info:
        print "Mean distance from center of us is " + str(int(np.mean(distances_from_center)))
        print "Median distance from center of us is " + str(int(np.median(distances_from_center)))
        print "Mean distance from nyc is " + str(int(np.mean(distances_from_nyc)))
        print "Median distance from nyc is " + str(int(np.median(distances_from_nyc)))
        print "Mean distance from la is " + str(int(np.mean(distances_from_la)))
        print "Median distance from la is " + str(int(np.median(distances_from_la)))

    return np.mean(distances), np.median(distances), acc_at_161

def lossbycoordinates(coordinates):
    if len(coordinates) != len(testUsers): 
        print "The number of test sample predictions is: " + str(len(coordinates))
        print "The number of test samples is: " + str(len(testUsers))
        print "fatal error!"
        sys.exit()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    U = testUsers.keys()
    for i in range(0, len(coordinates)):
        user = U[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        distances.append(distance(lat, lon, coordinates[i][0], coordinates[i][1]))
        
    # print "Average distance from class mean is " + str(averageMeanDistance)
    # print "Average distance from class median is " + str(averageMedianDistance)
    print "Mean distance is " + str(np.mean(distances))
    print "Median distance is " + str(np.median(distances))
    
    if loss == 'median':
        return np.median(distances)
    elif loss == 'mean':
        return np.mean(distances) 

  




def feature_extractor2(use_mention_dictionary=False, use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=1, max_df=1.0, BuildCostMatrices=False, vectorizer=None, stop_words=None, novectorization=False, vocab=None):
    '''
    read train, dev and test dictionaries and extract textual features using tfidfvectorizer.
    '''
    
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]

    

    print("%d categories" % len(categories))
    print()
    # split a training set and a test set
    Y_train = np.asarray([trainClasses[u] for u in U_train])
    Y_test = np.asarray([testClasses[u] for u in U_test])
    Y_dev = np.asarray([devClasses[u] for u in U_dev])
    
    if novectorization:
        X_train = None
        X_test = None
        X_dev = None
        feature_names = None
        return X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names  
    logging.info("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time.time()
    
    if vectorizer == None:    
        if use_mention_dictionary:
            print "using @ mention dictionary as vocab..."
            extract_mentions()
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=1, max_df=max_df, ngram_range=(1, 1), vocabulary=mentions, stop_words=stop_words)
        else:
            print "mindf: " + str(min_df) + " maxdf: " + str(max_df)
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words=stop_words, vocabulary=vocab, encoding=data_encoding)

    X_train = vectorizer.fit_transform([trainText[u] for u in U_train])
    feature_names = vectorizer.get_feature_names()
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    
    print("Extracting features from the dev dataset using the same vectorizer")
    t0 = time.time()
    X_dev = vectorizer.transform([devText[u] for u in U_dev])
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % X_dev.shape)
    print()

    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time.time()
    X_test = vectorizer.transform([testText[u] for u in U_test])
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
            
    return X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names

    
    

    
def classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity=10, DSExpansion=False, DSModification=False, compute_dev=False, report_verbose=False, clf=None, regul=0.00001, partitionMethod='median', penalty=None, fit_intercept=False, save_model=False, reload_model=False):
    model_dump_file = path.join(GEOTEXT_HOME, 'model-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '-' + str(regul) + '.pkl')
    top_features_file = path.join(GEOTEXT_HOME, 'topfeatures-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '-' + str(regul) + '.txt')
    compute_dev = True

    if clf == None:
        # clf = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l2', dual=False, tol=1e-3)
        # clf = linear_model.LogisticRegression(C=1.0, penalty='l2')
        # alpha = 0.000001
        clf = SGDClassifier(loss='log', alpha=regul, penalty=penalty, l1_ratio=0.9, learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=10, fit_intercept=fit_intercept)
        # clf = LabelPropagation(kernel='rbf', gamma=50, n_neighbors=7, alpha=1, max_iter=30, tol=0.001)
        # clf = LabelSpreading(kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=0.001)
        # clf = ensemble.AdaBoostClassifier()
        # clf = ensemble.RandomForestClassifier(n_jobs=10)
        # clf = MultiTaskLasso()
        # clf = ElasticNet()
        # clf = linear_model.Lasso(alpha = 0.1)
        
        # clf = SGDClassifier(loss, penalty, alpha, l1_ratio, fit_intercept, n_iter, shuffle, verbose, epsilon, n_jobs, random_state, learning_rate, eta0, power_t, class_weight, warm_start, rho, seed)
        # clf = linear_model.MultiTaskLasso(alpha=0.1)
        # clf = RidgeClassifier(tol=1e-2, solver="auto")
        # clf = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-2, class_weight=None, solver="auto")
        # clf = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
        # clf = Perceptron(n_iter=50)
        # clf = PassiveAggressiveClassifier(n_iter=50)
        # clf = KNeighborsClassifier(n_neighbors=10)
        # clf = NearestCentroid()
        # clf = MultinomialNB(alpha=.01)
    
    clf_requires_dense = False
    if clf_requires_dense:
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        X_dev = X_dev.toarray()
    model_reloaded = False
    if reload_model and path.exists(model_dump_file):
        print('loading a trained model from %s' % (model_dump_file))
        with open(model_dump_file, 'rb') as inf:
            clf = pickle.load(inf)
            model_reloaded = True
        print(clf)
    if not model_reloaded:
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time.time()
        clf.fit(X_train, Y_train)
        train_time = time.time() - t0
        print("train time: %0.3fs" % train_time)
        if hasattr(clf, 'coef_'):
            zero_count = (clf.coef_ == 0).sum()
            total_param_count = clf.coef_.shape[0] * clf.coef_.shape[1]
            zero_percent = int(100 * float(zero_count) / total_param_count)
            print('%d percent sparse' % (zero_percent))
            # if zero_percent > 50:
                # print('sparsifying clf.coef_ to free memory')
                # clf.sparsify()
        if save_model:
            print('dumpinng the model in %s' % (model_dump_file))
            with open(model_dump_file, 'wb') as outf:
                pickle.dump(clf, outf)
    if hasattr(clf, 'coef_'):
        non_zero_parameters = csr_matrix(clf.coef_).nnz
    else:
        non_zero_parameters = 0
    plot_matrix = False
    if plot_matrix and hasattr(clf, 'coef_'):
        print('plotting the first 100 features of the weight matrix...')
        plt.close()
        plt.pcolormesh(clf.coef_[:, 0:100], cmap='RdBu', vmin=np.min(clf.coef_[:, 0:100]), vmax=np.max(clf.coef_[:, 0:100]))
        plt.savefig(path.join(GEOTEXT_HOME, 'matrix.pdf'), format='pdf')
        plt.close()
    report_verbose = True
    if compute_dev:
        devPreds = clf.predict(X_dev)
        if hasattr(clf, 'predict_proba'):
            devProbs = clf.predict_proba(X_dev)
        else:
            devProbs = None

    save_vocabulary = False
    if save_vocabulary:
        rows, cols = np.nonzero(clf.coef_)
        del rows
        vocab = feature_names[cols]
        vocab_file = path.join(GEOTEXT_HOME, 'vocab.pkl')
        with open(vocab_file, 'wb') as outf:
            pickle.dump(vocab, outf)
    abod = False
    print "predicting test labels"  
    t0 = time.time()
    preds = clf.predict(X_test)
    # scores = clf.decision_function(X_test)
    if hasattr(clf, 'predict_proba'):
        testProbs = clf.predict_proba(X_test)
    else:
        testProbs = None
    probs = None
    # print preds.shape
    test_time = time.time() - t0
    print("test time: %0.3fs" % test_time)
    

    report_verbose = False
    if report_verbose:
        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
            print("top 10 keywords per class:")
            with codecs.open(top_features_file, 'w', encoding='utf-8') as outf:
                for i, category in enumerate(categories):
                    top10 = np.argsort(clf.coef_[i])[-50:]
                    # print("%s: %s" % (category, " ".join(feature_names[top10])))
                    outf.write(category + ": " + " ".join(feature_names[top10]) + '\n')


    
    print "test results"
    meanTest, medianTest, acc_at_161_test = loss(preds, U_test, save=True)
    meanDev = -1
    medianDev = -1
    if compute_dev:
        print "development results"
        meanDev, medianDev, acc_at_161_dev = loss(devPreds, U_dev, save_results_bucket_size=True, results_key='lr')
    dump_preds = True
    if dump_preds:
        result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '.pkl')
        print "dumping preds (preds, devPreds, U_test, U_dev, testProbs, devProbs) in " + result_dump_file
        with open(result_dump_file, 'wb') as outf:
            pickle.dump((preds, devPreds, U_test, U_dev, testProbs, devProbs), outf)
    # evaluate(preds,U_test, categories, None)
    # abod(probs, preds, U_test)
    return preds, probs, U_test, meanTest, medianTest, acc_at_161_test, meanDev, medianDev, acc_at_161_dev, non_zero_parameters
   
# classify()









def initialize(partitionMethod, granularity, write=False, readText=True, reload_init=False, regression=False):    

    reload_file = path.join(GEOTEXT_HOME + '/init_' + DATASETS[DATASET_NUMBER - 1] + '_' + partitionMethod + '_' + str(BUCKET_SIZE) + '.pkl')

    logging.info('reading (user_info.) train, dev and test file and building trainUsers, devUsers and testUsers with their locations')
    users(trainfile, 'train', write, readText=readText)
    users(devfile, 'dev', write, readText=readText)
    users(testfile, 'test', write, readText=readText)
    logging.info("the number of train" + " users is " + str(len(trainUsers)))
    logging.info("the number of test" + " users is " + str(len(testUsers)))
    logging.info("the number of dev" + " users is " + str(len(devUsers)))

    if not regression:
        create_directories(granularity, partitionMethod, write)  
        print('writing init info in %s' % (reload_file))
        with open(reload_file, 'wb') as outf:
            pickle.dump((classLatMean, classLonMedian, classLatMedian, classLonMean, userLocation, categories, trainUsers, trainClasses, testUsers, testClasses, devUsers, devClasses), outf)
    else:
        logging.info('Not discretising locations as regression option is on!')
    logging.info("initialization finished")


def factorize(X, transformees, factorizer=None):
    if factorizer is None:
        factorizer = DictionaryLearning(n_components=100, alpha=1, max_iter=100, tol=1e-8, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=30, code_init=None, dict_init=None, verbose=True, split_sign=None, random_state=None)
    else:
        print factorizer
    # dic_learner = DictionaryLearning(n_components=100, alpha=1, max_iter=100, tol=1e-8, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=30, code_init=None, dict_init=None, verbose=True, split_sign=None, random_state=None)
    # dic_learner = MiniBatchDictionaryLearning(n_components=20, alpha=1, n_iter=100, fit_algorithm='lars', n_jobs=30, batch_size=1000, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=True, split_sign=True, random_state=None)
    # dic_learner = PCA(n_components=500)
    
    
    if sparse.issparse(X):
        X = X.toarray()
    factorizer.fit(X)
    results = []
    for feature_matrix in transformees:
        if sparse.issparse(feature_matrix):
            feature_matrix = feature_matrix.toarray()
        feature_matrix = factorizer.transform(feature_matrix)
        results.append(feature_matrix)
    return results

def asclassification(granularity, partitionMethod, use_mention_dictionary=False, use_sparse_code=False, penalty=None, fit_intercept=False, norm=None, binary=False, sublinear=False, factorizer=None, read_vocab=False, use_idf=True):

    if read_vocab:
        vocab_file = path.join(GEOTEXT_HOME, 'vocab.pkl')
        with open(vocab_file, 'rb') as inf:
            vocab = pickle.load(inf)
            vocab = list(set(vocab))
    else:
        vocab = None
    stops = 'english'
    # partitionLocView(granularity=granularity, partitionMethod=partitionMethod)
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor2(norm=norm, use_mention_dictionary=use_mention_dictionary, min_df=10, max_df=0.2, stop_words=stops, binary=binary, sublinear_tf=sublinear, vocab=vocab, use_idf=use_idf)    
    if use_sparse_code:
        print("using a matrix factorization technique to learn a better representation of users...")
        sparse_coded_dump = path.join(GEOTEXT_HOME, 'sparse_coded.pkl')
        if os.path.exists(sparse_coded_dump) and DATASET_NUMBER != 1:
            with open(sparse_coded_dump, 'rb') as inf:
                X_train, X_dev, X_test = pickle.load(inf)
        else:
            X_train, X_dev, X_test = factorize(X_train, transformees=[X_train, X_dev, X_test], factorizer=factorizer)
            save_sparse = False
            if save_sparse:
                with open(sparse_coded_dump, 'wb') as inf:
                    pickle.dump((X_train, X_dev, X_test), inf)
    
    best_dev_acc = -1
    best_dev_mean = -1
    best_dev_median = -1
    best_regul = -1
    best_test_preds = None
    regul_acc = {}
    regul_nonzero = {}
    
    
    if DATASET_NUMBER == 1:
        reguls_coefs = [5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 4e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-2, 1e-2]
    elif DATASET_NUMBER == 2:
        reguls_coefs = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
        reguls_coefs = [2 ** a for a in xrange(-27, -3, 2)]
        if BUCKET_SIZE <= 1024:
            reguls_coefs = [2 ** -21]
        else:
            reguls_coefs = [2 ** -19]
    elif DATASET_NUMBER == 3:
        reguls_coefs = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
    if penalty == 'none':
        reguls_coefs = [1e-200]
    # for regul in reguls_coefs:
    for regul in [reguls[DATASET_NUMBER - 1]]:
        preds, probs, U_test, meanTest, medianTest, acc_at_161_test, meanDev, medianDev, acc_at_161_dev, non_zero_parameters = classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity=granularity, regul=regul, partitionMethod=partitionMethod, penalty=penalty, fit_intercept=fit_intercept, reload_model=False)
        regul_acc[regul] = acc_at_161_dev
        regul_nonzero[regul] = non_zero_parameters
        if acc_at_161_dev > best_dev_acc:
            best_dev_acc = acc_at_161_dev
            best_regul = regul
            best_test_preds = preds
            best_dev_mean = meanDev
            best_dev_median = medianDev
            
    print('The best regul_coef is %e %f' % (best_regul, best_dev_acc))
    t_mean, t_median, t_acc = loss(best_test_preds, U_test)
    
    return t_mean, t_median, t_acc, best_dev_mean, best_dev_median, best_dev_acc
    # return preds, probs, U_test, meanTest, medianTest, acc_at_161_test, meanDev, medianDev, acc_at_161_dev


def locationStr2Float(locationStr):
    latlon = locationStr.split(',')
    lat = float(latlon[0])
    lon = float(latlon[1])
    return lat, lon






def extract_mentions(k=0, addTest=False, addDev=False):
    if addTest and addDev:
        print "addTest and addDev can not be True in the same time"
        sys.exit(0)
    print "extracting mention information from text"
    global mentions
    # if it is there load it and return
    mention_file_address = path.join(GEOTEXT_HOME, 'mentions.pkl')
    if addDev:
        mention_file_address = mention_file_address + '.dev'
    RELOAD_MENTIONS = True
    if RELOAD_MENTIONS:
        if os.path.exists(mention_file_address):
            print "reading mentions from pickle"
            with open(mention_file_address, 'rb') as inf:
                mentions = pickle.load(inf)
                return
    text = ''
    # for user in trainUsers:
    #    text += userText[user].lower()
    if addTest:
        text = ' '.join(trainText.values() + testText.values())
    if addDev:
        text = ' '.join(trainText.values() + devText.values())
    if not addTest and not addDev:
        text = ' '.join(trainText.values())
    # text = text.lower()
    '''
    if data_encoding in ['utf-8']:
        text = strip_accents_unicode(text)
    elif data_encoding in ['latin', 'latin1']:
        text = strip_accents_ascii(text)
    '''
    token_pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern = re.compile(token_pattern)
    print "finding mentions"
    mentionsList = [word.lower() for word in token_pattern.findall(text)]
    print "building the counter"
    mentionsDic = Counter(mentionsList)
    print "frequency thresholding"
    if k > 0:
        mentions = [word for word in mentionsDic if mentionsDic[word] > k]
    else:
        mentions = mentionsDic.keys()

    with open(mention_file_address, 'wb') as outf:
        print "writng mentions to pickle"
        pickle.dump(mentions, outf)

def prepare_adsorption_data_collapsed(DEVELOPMENT=False, text_prior='none', CELEBRITY_THRESHOLD=100000, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=False, partitionMethod='median'):
    MULTI_LABEL = False

    dongle_nodes = None
    dongle_preds = None
    dongle_probs = None
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    text_str = ''
    if text_prior != 'none':
        text_str = '.' + text_prior
    weighted_str = ''
    if DIRECT_GRAPH_WEIGHTED:
        weighted_str = '.weighted'
    
    celebrityStr = str(CELEBRITY_THRESHOLD)

    
    # split a training set and a test set
    Y_train = np.asarray([trainClasses[u] for u in U_train])
    # Y_test = np.asarray([testClasses[u] for u in U_test])
    # Y_dev = np.asarray([devClasses[u] for u in U_dev])

    
    if DEVELOPMENT:
        devStr = '.dev'
    else:
        devStr = ''

    
    if text_prior != 'none':
        logging.info("tex prior is " + text_prior)
        # read users and predictions
        result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '.pkl')
        t_preds, d_preds, t_users, d_users, t_probs, d_probs = None, None, None, None, None, None
        logging.info("reading the text learner results from " + result_dump_file)
        with open(result_dump_file, 'rb') as inf:
            t_preds, d_preds, t_users, d_users, t_probs, d_probs = pickle.load(inf)

        if DEVELOPMENT:
            dongle_nodes = d_users
            dongle_preds = d_preds
            dongle_probs = d_probs
            logging.info("text dev results:")
            loss(d_preds, d_users)
        else:
            dongle_nodes = t_users
            dongle_preds = t_preds
            dongle_probs = t_probs
            logging.info("text test results:")
            loss(t_preds, t_users)
              

    id_user_file = path.join(GEOTEXT_HOME, 'id_user_' + partitionMethod + '_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing id_user in " + id_user_file)
    with codecs.open(id_user_file, 'w', 'ascii') as outf:
        for i in range(0, len(U_train)):
            outf.write(str(i) + '\t' + U_train[i] + '\n')
        if DEVELOPMENT:
            for i in range(0, len(U_dev)):
                outf.write(str(i + len(U_train)) + '\t' + U_dev[i] + '\n')
        else:
            for i in range(0, len(U_test)):
                outf.write(str(i + len(U_train)) + '\t' + U_test[i] + '\n')

    seed_file = path.join(GEOTEXT_HOME, 'seeds_' + partitionMethod + '_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing seeds in " + seed_file)
    with codecs.open(seed_file, 'w', 'ascii') as outf:
        for i in range(0, len(U_train)):
            outf.write(str(i) + '\t' + str(Y_train[i]) + '\t' + '1.0' + '\n')

        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                # w = np.max(dongle_probs[i])
                w = 1
                outf.write(str(i + len(U_train)) + '.T' + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'direct':
            w = np.max(dongle_probs[i])
            outf.write(str(i + len(U_train)) + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'backoff':
            pass 

    doubles = 0
    double_nodes = []
    trainIdx = range(len(U_train))
    trainUsersLowerDic = dict(zip(U_train, trainIdx))
    if DEVELOPMENT:
        devStr = '.dev'
        for i in range(0, len(U_dev)):
            u = U_dev[i]
            if u in trainUsersLowerDic:
                U_dev[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_unknown = U_dev
        u_text_unknown = devText
    else:
        for i in range(0, len(U_test)):
            u = U_test[i]
            if u in trainUsersLowerDic:
                U_test[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_text_unknown = testText
        u_unknown = U_test
    if text_prior != 'none':
        assert(len(u_unknown) == len(dongle_nodes)), 'the number of text/dev users is not equal to the number of text predictions.'
    U_all = U_train + u_unknown 

    logging.info("The number of test users found in train users is " + str(doubles))
    logging.info("Double users: " + str(double_nodes))
    vocab_cnt = len(U_all)
    idx = range(vocab_cnt)
    node_id = dict(zip(U_all, idx))
    # data and indices of a coo matrix to be populated
    coordinates = Counter()
    
    # for node, id in node_id.iteritems():
    #    node_lower_id[node.lower()] = id
    assert (len(node_id) == len(U_train) + len(u_unknown)), 'number of unique users is not eq u_train + u_test'
    logging.info("the number of nodes is " + str(vocab_cnt))
    logging.info("Adding the direct relationships...")
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    mention_users = defaultdict(Counter)
    directly_connected_users = set()
    l = len(trainText)
    tenpercent = l / 10
    i = 1
    for user, text in trainText.iteritems():
        user_id = node_id[user]
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        mentions = [u.lower() for u in token_pattern1.findall(text)] 
        mentionDic = Counter(mentions)
        for mention, freq in mentionDic.iteritems():
            # check if mention is a user node
            mention_id = node_id.get(mention, -1)
            if mention_id not in [-1, user_id]:    
                if user_id < mention_id:
                    coordinates[(user_id, mention_id)] += freq
                    directly_connected_users.add(user_id)
                    directly_connected_users.add(mention_id)
                elif mention_id < user_id:
                    coordinates[(mention_id, user_id)] += freq
                    directly_connected_users.add(user_id)
                    directly_connected_users.add(mention_id)

                
            mention_users[mention][user_id] += freq
    
    
    
    logging.info("adding the eval graph")
    for user, text in u_text_unknown.iteritems():
        user_id = node_id[user]
        mentions = [u.lower() for u in token_pattern1.findall(text)]
        mentionDic = Counter(mentions)
        for mention, freq in mentionDic.iteritems():
            mention_id = node_id.get(mention, -1)
            if mention_id != -1:
                if mention_id != user_id:
                    if user_id < mention_id:
                        coordinates[(user_id, mention_id)] += freq
                        directly_connected_users.add(user_id)
                        directly_connected_users.add(mention_id)
                    elif mention_id < user_id:
                        coordinates[(mention_id, user_id)] += freq
                        directly_connected_users.add(user_id)
                        directly_connected_users.add(mention_id)

            mention_users[mention][user_id] += freq
    
    
    
    logging.info("Direct Relationships: " + str(len(coordinates)))
    logging.info("Directly related users: " + str(len(directly_connected_users)) + " percent: " + str(100.0 * len(directly_connected_users) / len(U_all)))
    logging.info("Adding the collapsed indirect relationships...")

    l = len(mention_users)
    tenpercent = l / 10
    i = 1
    celebrities_count = 0
    for mention, user_ids in mention_users.iteritems():
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        if len(user_ids) > CELEBRITY_THRESHOLD:
            celebrities_count += 1
            continue
        
        for user_id1, freq1 in user_ids.iteritems():
            for user_id2, freq2 in user_ids.iteritems():
                if user_id1 < user_id2:
                    coordinates[(user_id1, user_id2)] += (freq1 + freq2)

    # free memory by deleting mention_users
    del mention_users
    logging.info("The number of celebrities is " + str(celebrities_count) + " .")
    logging.info("The number of edges is " + str(len(coordinates)))
    l = len(coordinates)
    tenpercent = l / 10
    input_graph_file = path.join(GEOTEXT_HOME, 'input_graph_' + partitionMethod + '_' + str(BUCKET_SIZE) + '_' + celebrityStr + devStr + text_str + weighted_str)
    logging.info("writing the input_graph in " + input_graph_file)
    with codecs.open(input_graph_file, 'w', 'ascii', buffering=pow(2, 6) * pow(2, 20)) as outf:
        i = 1
        for nodes, w in coordinates.iteritems():
            xindx, yindx = nodes
            if not DIRECT_GRAPH_WEIGHTED:
                w = 1.0
            if i % tenpercent == 0:
                logging.info("processing " + str(10 * i / tenpercent) + "%")
            i += 1
            outf.write(str(xindx) + '\t' + str(yindx) + '\t' + str(w) + '\n')
            if build_networkx_graph:
                mention_graph.add_edge(xindx, yindx, attr_dict={'w':w})
        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                confidence = np.max(dongle_probs[i])
                outf.write(str(i + len(U_train)) + '.T' + '\t' + str(i + len(U_train)) + '\t' + str(confidence) + '\n')
        elif text_prior == 'direct':
            pass
        elif text_prior == 'backoff':
            pass
    
    u1s = [a for a, b in coordinates]
    u2s = [b for a, b in coordinates]
    us = set(u1s + u2s)
    logging.info("the number of disconnected users is " + str(len(U_all) - len(us)))
    logging.info("the number of disconnected test users is " + str(len(u_unknown) - len([a for a in us if a >= len(U_train)])))
    output_file = path.join(GEOTEXT_HOME, 'label_prop_output_' + DATASETS[DATASET_NUMBER - 1] + '_' + partitionMethod + '_' + str(BUCKET_SIZE) + '_' + str(celeb_threshold) + devStr + text_str + weighted_str)
    logging.info("output file: " + output_file)
    # logging.info(str(disconnected_us))

def prepare_adsorption_data_collapsed_networkx(DEVELOPMENT=False, text_prior='none', CELEBRITY_THRESHOLD=100000, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=True, partitionMethod='median', postfix='.nx'):
    global mentions

    MULTI_LABEL = False

    dongle_nodes = None
    dongle_preds = None
    dongle_probs = None
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    text_str = ''
    if text_prior != 'none':
        text_str = '.' + text_prior
    weighted_str = ''
    if DIRECT_GRAPH_WEIGHTED:
        weighted_str = '.weighted'
    
    celebrityStr = str(CELEBRITY_THRESHOLD)
    mention_graph = nx.Graph()

    Y_train = np.asarray([trainClasses[u] for u in U_train])

    
    if DEVELOPMENT:
        devStr = '.dev'
    else:
        devStr = ''



    
    if text_prior != 'none':
        logging.info("tex prior is " + text_prior)
        # read users and predictions
        result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '.pkl')
        t_preds, d_preds, t_users, d_users, t_probs, d_probs = None, None, None, None, None, None
        logging.info("reading the text learner results from " + result_dump_file)
        with open(result_dump_file, 'rb') as inf:
            t_preds, d_preds, t_users, d_users, t_probs, d_probs = pickle.load(inf)

        if DEVELOPMENT:
            dongle_nodes = d_users
            dongle_preds = d_preds
            dongle_probs = d_probs
            logging.info("text dev results:")
            loss(d_preds, d_users)
        else:
            dongle_nodes = t_users
            dongle_preds = t_preds
            dongle_probs = t_probs
            logging.info("text test results:")
            loss(t_preds, t_users)
              
    id_user_file = path.join(GEOTEXT_HOME, 'id_user_' + partitionMethod + '_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing id_user in " + id_user_file)
    with codecs.open(id_user_file, 'w', 'ascii') as outf:
        for i in range(0, len(U_train)):
            outf.write(str(i) + '\t' + U_train[i] + '\n')
        if DEVELOPMENT:
            for i in range(0, len(U_dev)):
                outf.write(str(i + len(U_train)) + '\t' + U_dev[i] + '\n')
        else:
            for i in range(0, len(U_test)):
                outf.write(str(i + len(U_train)) + '\t' + U_test[i] + '\n')

    seed_file = path.join(GEOTEXT_HOME, 'seeds_' + partitionMethod + '_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing seeds in " + seed_file)
    with codecs.open(seed_file, 'w', 'ascii') as outf:
        for i in range(0, len(U_train)):
            outf.write(str(i) + '\t' + str(Y_train[i]) + '\t' + '1.0' + '\n')

        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                # w = np.max(dongle_probs[i])
                w = 1
                outf.write(str(i + len(U_train)) + '.T' + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'direct':
            w = np.max(dongle_probs[i])
            outf.write(str(i + len(U_train)) + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'backoff':
            pass 

    doubles = 0
    double_nodes = []
    trainIdx = range(len(U_train))
    trainUsersLowerDic = dict(zip(U_train, trainIdx))
    if DEVELOPMENT:
        devStr = '.dev'
        for i in range(0, len(U_dev)):
            u = U_dev[i]
            if u in trainUsersLowerDic:
                U_dev[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_unknown = U_dev
        u_text_unknown = devText
    else:
        for i in range(0, len(U_test)):
            u = U_test[i]
            if u in trainUsersLowerDic:
                U_test[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_text_unknown = testText
        u_unknown = U_test
    if text_prior != 'none':
        assert(len(u_unknown) == len(dongle_nodes)), 'the number of text/dev users is not equal to the number of text predictions.'
    
    
    
    U_all = U_train + u_unknown 
    logging.info("The number of test users found in train users is " + str(doubles))
    logging.info("Double users: " + str(double_nodes))
    vocab_cnt = len(U_all)
    idx = range(vocab_cnt)
    node_id = dict(zip(U_all, idx))
    idx = list(idx)
    mention_graph.add_nodes_from(idx)
    # for node, id in node_id.iteritems():
    #    node_lower_id[node.lower()] = id
    assert (len(node_id) == len(U_train) + len(u_unknown)), 'number of unique users is not eq u_train + u_test'
    logging.info("the number of nodes is " + str(vocab_cnt))
    logging.info("Adding the direct relationships...")
    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    l = len(trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    mention_graph.add_nodes_from(node_id.values())
    for user, text in trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user_id] 
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not DIRECT_GRAPH_WEIGHTED:
                freq = 1
            if mention_graph.has_edge(user_id, mention):
                mention_graph[user_id][mention]['weight'] += freq
                # mention_graph[mention][user]['weight'] += freq/2.0
            else:
                mention_graph.add_edge(user_id, mention, weight=freq)
                # mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in u_text_unknown.iteritems():
        user_id = node_id[user]
        mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user_id]
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not DIRECT_GRAPH_WEIGHTED:
                freq = 1
            if mention_graph.has_edge(user_id, mention):
                mention_graph[user_id][mention]['weight'] += freq
                # mention_graph[mention][user]['weight'] += freq/2.0
            else:
                mention_graph.add_edge(user_id, mention, weight=freq)
                # mention_graph.add_edge(mention, user, weight=freq/2.0)  
    
    
    celebrities = []
    remove_celebrities = True
    if remove_celebrities:
        nodes = mention_graph.nodes_iter()
        for node in nodes:
            nbrs = mention_graph.neighbors(node)
            if len(nbrs) > CELEBRITY_THRESHOLD:
                celebrities.append(node)
        celebrities = [c for c in celebrities if c not in idx]
        logging.info("found %d celebrities with celebrity threshold %d" % (len(celebrities), CELEBRITY_THRESHOLD))
        for celebrity in celebrities:
                mention_graph.remove_node(celebrity)
    
    project_to_main_users = True
    if project_to_main_users:
        # mention_graph = bipartite.overlap_weighted_projected_graph(mention_graph, main_users, jaccard=False)
        mention_graph = collaboration_weighted_projected_graph(mention_graph, idx, weight_str=None, degree_power=1, caller='mad')
        # mention_graph = collaboration_weighted_projected_graph(mention_graph, main_users, weight_str='weight')
        # mention_graph = bipartite.projected_graph(mention_graph, main_users)
    connected_nodes = set()
    l = mention_graph.number_of_edges()
    logging.info("The number of edges is " + str(l))
    tenpercent = l / 10
    input_graph_file = path.join(GEOTEXT_HOME, 'input_graph_' + partitionMethod + '_' + str(BUCKET_SIZE) + '_' + celebrityStr + devStr + text_str + weighted_str + postfix)
    logging.info("writing the input_graph in " + input_graph_file)
    with codecs.open(input_graph_file, 'w', 'ascii', buffering=pow(2, 6) * pow(2, 20)) as outf:
        i = 1
        for edge in mention_graph.edges_iter(nbunch=None, data=True):
            u, v , data = edge
            connected_nodes.add(u)
            connected_nodes.add(v)
            w = data['weight']
            if not DIRECT_GRAPH_WEIGHTED:
                w = 1.0
            if i % tenpercent == 0:
                logging.info("processing " + str(10 * i / tenpercent) + "%")
            i += 1
            outf.write(str(u) + '\t' + str(v) + '\t' + str(w) + '\n')
        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                confidence = np.max(dongle_probs[i])
                outf.write(str(i + len(U_train)) + '.T' + '\t' + str(i + len(U_train)) + '\t' + str(confidence) + '\n')
        elif text_prior == 'direct':
            pass
        elif text_prior == 'backoff':
            pass
    
    
    us = [a for a in idx if a in connected_nodes]
    logging.info("the number of disconnected users is " + str(len(U_all) - len(us)))
    logging.info("the number of disconnected test users is " + str(len(u_unknown) - len([a for a in us if a >= len(U_train)])))
    output_file = path.join(GEOTEXT_HOME, 'label_prop_output_' + DATASETS[DATASET_NUMBER - 1] + '_' + partitionMethod + '_' + str(BUCKET_SIZE) + '_' + str(celeb_threshold) + devStr + text_str + weighted_str)
    logging.info("output file: " + output_file)
    # logging.info(str(disconnected_us))
    


def ideal_network_errors():
    graph_file_address = path.join(GEOTEXT_HOME, 'direct_graph')
    if os.path.exists(graph_file_address):
        print "reading netgraph from pickle"
        with open(graph_file_address, 'rb') as inf:
            netgraph, trainUsers, testUsers = pickle.load(inf)
    ideal_distances = []
    acc161 = 0
    tenpercent = len(testUsers) / 10
    i = 0
    for utest, uloc in testUsers.iteritems():
        i += 1
        if i % tenpercent == 0:
            print str(100 * i / len(testUsers))
        lat1, lon1 = locationStr2Float(uloc)
        dists = []
        for utrain, utrainloc in trainUsers.iteritems():
            lat2, lon2 = locationStr2Float(utrainloc)
            d = distance(lat1, lon1, lat2, lon2)
            dists.append(distance(lat1, lon1, lat2, lon2))
            if d < 1:
                break
        minDist = min(dists)
        if minDist < 161:
            acc161 += 1
        ideal_distances.append(minDist)
    print "distance number" + str(len(ideal_distances))
    print "mean " + str(np.mean(ideal_distances))
    print "median " + str(np.median(ideal_distances))
    print "Acc @ 161 " + str((acc161 + 0.0) / len(ideal_distances))
    


                
def LP(weighted=True, prior='none', normalize_edge=False, remove_celebrities=False, dev=False, node_order='l2h', remove_mentions_with_degree_one=True):
    '''
    Run label propagation over real-valued coordinates of the @-mention graph.
    The coordinates of the training users are kept unchanged and the coordinates
    of other users are updated to the median of their neighbours.
    '''
    if dev:
        evalText = devText
        evalUsers = devUsers
    else:
        evalText = testText
        evalUsers = testUsers
    save_gr = False
    verbose = False
    mention_graph = nx.Graph()
    graph_file_address = path.join(GEOTEXT_HOME, 'direct_graph.graphml')
    U_all = trainUsers.keys() + evalUsers.keys()
    assert len(U_all) == len(trainUsers) + len(evalUsers), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    
    print('weighted=%s and prior=%s' % (weighted, prior))
    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    netgraph = {}
    l = len(trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    mention_graph.add_nodes_from([(u, {'train':True, 'loc':l}) for u, l in trainUsers.iteritems()])
    mention_graph.add_nodes_from([(u, {'test':True, 'loc':l}) for u, l in evalUsers.iteritems()])
    for user, text in trainText.iteritems():    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        mentions = [u.lower() for u in token_pattern1.findall(text)] 
        mentions = [m for m in mentions if m != user]
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if mention_graph.has_edge(user, mention):
                mention_graph[user][mention]['weight'] += freq
            else:
                mention_graph.add_edge(user, mention, weight=freq)   
        
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        mentions = [u.lower() for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user]
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if mention_graph.has_edge(user, mention):
                mention_graph[user][mention]['weight'] += freq
            else:
                mention_graph.add_edge(user, mention, weight=freq)  
    

    lp_graph = mention_graph          
    trainUsersLower = {}
    evalUsersLower = {}
    trainLats = []
    trainLons = []
    node_location = {}
    dongle_nodes = []
    text_preds = {}
    
    dongle = False
    backoff = False
    text_direct = False
    if prior != 'none':
        if prior == 'backoff':
            backoff = True
        elif prior == 'prior':
            text_direct = True
        elif prior == 'dongle':
            dongle = True
        prior_file_path = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '.pkl')
        print "reading prior text-based locations from " + prior_file_path
        if os.path.exists(prior_file_path):
            with open(prior_file_path, 'rb') as inf:
                preds, devPreds, U_test, U_dev, testProbs, devProbs = pickle.load(inf)
                if dev:
                    preds = devPreds
                    U_test = U_dev
                    testProbs = devProbs
                test_confidences = testProbs[np.arange(0, preds.shape[0]), preds]
                loss(preds=preds, U_test=U_test)
                if dongle:
                    mention_graph.add_nodes_from([u + '.dongle' for u in U_test])
                user_index = 0   
                for user, pred in zip(U_test, preds):
                    lat = classLatMedian[str(pred)]
                    lon = classLonMedian[str(pred)]
                    if backoff:
                        text_preds[user] = (lat, lon)
                    if dongle:
                        dongle_node = user + '.dongle'
                        w = test_confidences[user_index]
                        mention_graph.add_edge(dongle_node, user, weight=w)
                        node_location[dongle_node] = (lat, lon)
                        dongle_nodes.append(dongle_node)
                    elif text_direct:
                        node_location[user] = (lat, lon)
                    user_index += 1
        else:
            print "prior file not found."
    for user, loc in trainUsers.iteritems():
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)
        trainUsersLower[user] = (lat, lon)
        node_location[user] = (lat, lon)
        
    for user, loc in evalUsers.iteritems():
        lat, lon = locationStr2Float(loc)
        evalUsersLower[user] = (lat, lon)
    
    lp_graph = evalUsersLower
    print "the number of train nodes is " + str(len(trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    medianLat = np.median(trainLats)
    medianLon = np.median(trainLons)

    # remove celebrities from the graph
    
    remove_betweeners = False
    if remove_betweeners:
        print("computing betweenness centrality of all nodes, takes a long time, sorry!")
        scores = nx.betweenness_centrality(mention_graph, weight='weight')
        i = 0
        percent_5 = len(scores) / 20 
        for w in sorted(scores, key=scores.get, reverse=True):
            i += 1
            if i < percent_5:
                mention_graph.remove_node(w)
    
    
    celebrity_threshold = celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = mention_graph.nodes_iter()
        for node in nodes:
            nbrs = mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in evalUsersLower and node not in trainUsersLower:
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
            mention_graph.remove_node(celebrity)
 
    if remove_mentions_with_degree_one:
        mention_nodes = set(mention_graph.nodes()) - set(node_id.values())
        mention_degree = mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = [node for node, degree in mention_degree.iteritems() if degree < 2]
        logging.info('found ' + str(len(one_degree_non_target)) + ' mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            mention_graph.remove_node(node)
    print "finding unlocated nodes"
    if node_order:
        node_degree = mention_graph.degree()
        # sort node_degree by value
        if node_order == 'h2l':
            reverse_order = True
        else:
            reverse_order = False
        nodes = sorted(node_degree, key=node_degree.get, reverse=reverse_order)
        if node_order == 'random':
            random.shuffle(nodes)
        # nodes_unknown = [node for node in mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]
        nodes_unknown = [node for node in nodes if node not in trainUsersLower and node not in dongle_nodes]
    else:
        nodes_unknown = [node for node in mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]

    # find the cycles with 3 nodes and increase their edge weight
    increase_cyclic_edge_weights = False
    if increase_cyclic_edge_weights:
        increase_coefficient = 2
        cycls_3 = [c for c in list(nx.find_cliques(mention_graph)) if len(c) > 2]
        print(str(len(cycls_3)) + ' triangles in the graph.')
        # cycls_3 = [c for c in nx.cycle_basis(mention_graph) if len(c)==3]
        for c_3 in cycls_3:
            mention_graph[c_3[0]][c_3[1]]['weight'] *= increase_coefficient
            mention_graph[c_3[0]][c_3[2]]['weight'] *= increase_coefficient
            mention_graph[c_3[1]][c_3[2]]['weight'] *= increase_coefficient
        del cycls_3
    
    # remove (or decrease the weights of) the edges between training nodes which are very far from each other
    remove_inconsistent_edges = False
    if remove_inconsistent_edges:
        max_acceptable_distance = 161
        num_nodes_removed = 0
        edges = mention_graph.edges()
        edges = [(a, b) for a, b in edges if a in node_location and b in node_location]
        for node1, latlon1 in node_location.iteritems():
            for node2, latlon2 in node_location.iteritems():
                lat1, lon1 = latlon1
                lat2, lon2 = latlon2
                dd = distance(lat1, lon1, lat2, lon2)
                if dd > max_acceptable_distance:
                    if ((node1, node2) in edges or (node2, node1) in edges):
                        try:
                            mention_graph.remove_edge(node1, node2)
                            num_nodes_removed += 1
                        except:
                            pass
        print(str(num_nodes_removed) + ' edges removed from the graph') 
                
    use_shortest_paths = False
    shortest_paths = {}
    if use_shortest_paths:
        shortest_paths = nx.all_pairs_shortest_path_length(mention_graph, cutoff=3)
    

    
    logging.info("Edge number: " + str(mention_graph.number_of_edges()))
    logging.info("Node number: " + str(mention_graph.number_of_nodes()))
    
    converged = False
    print "weighted " + str(weighted)
    max_iter = 5
    iter_num = 1
    print "iterating with max_iter = " + str(max_iter)

    # if selfish = True, the nodes location would be added to that of its neighbours and then the median is computed. (it didn't improve the results on cmu)
    selfish = False
    while not converged:
        if node_order == 'random':
            random.shuffle(nodes)
        isolated_users = set()
        print "iter: " + str(iter_num)
        located_nodes_count = len(node_location)
        print str(located_nodes_count) + " nodes have location"
        for node in nodes_unknown:
            nbrs = mention_graph[node]
            nbrlats = []
            nbrlons = []
            nbr_edge_weights = []
            
            if selfish:
                if node in node_location:
                    self_lat, self_lon = node_location[node]
                    nbrlats.append(self_lat)
                    nbrlons.append(self_lon)
                
            for nbr in nbrs:
                if nbr in node_location:
                    lat, lon = node_location[nbr]
                    edge_weight = mention_graph[node][nbr]['weight']
                    nbrlats.append(lat)
                    nbrlons.append(lon)
                    if normalize_edge and weighted:
                        edge_weight_normalized = float(edge_weight) / (mention_graph.degree(nbr) * mention_graph.degree(node))
                        nbr_edge_weights.append(edge_weight_normalized)
                    # elif not weighted:
                    #    nbr_edge_weights.append(1)
                    else:
                        nbr_edge_weights.append(edge_weight)
                    
            if use_shortest_paths:
                if node in shortest_paths:
                    community_nbrs = shortest_paths[node]
                    for nbr, path_length in community_nbrs.iteritems():
                        if path_length > 1:
                            if nbr in node_location:
                                lat, lon = node_location[nbr]
                                nbrlats.append(lat)
                                nbrlons.append(lon)
            if len(nbrlons) > 0:
                nbr_median_lat, nbr_median_lon = weighted_median(nbrlats, nbr_edge_weights), weighted_median(nbrlons, nbr_edge_weights)
                node_location[node] = (nbr_median_lat, nbr_median_lon)

        iter_num += 1
        if iter_num == max_iter:
            converged = True
        
        if len(node_location) == located_nodes_count:
            print "converged. No new nodes added in this iteration."
            # converged = True
        distances = []
        isolated = 0
        for user, loc in evalUsersLower.iteritems():
            lat, lon = loc
            if user not in node_location:
                isolated += 1
                isolated_users.add(user)
            if backoff and user in isolated_users:
                predicted_lat, predicted_lon = text_preds[user]
            else:
                predicted_lat, predicted_lon = node_location.get(user, (medianLat, medianLon))
            dist = distance(lat, lon, predicted_lat, predicted_lon)
            distances.append(dist)
        current_median = np.median(distances)
        current_mean = np.mean(distances)
        current_acc = 100 * len([d for d in distances if d < 161]) / float(len(distances))

        print "mean: " + str(int(current_mean))
        print "median:" + str(int(current_median))
        print "Acc@161:" + str(current_acc)
        print "isolated test users are " + str(isolated) + " out of " + str(len(distances))

    return current_mean, current_median, current_acc
    

def LP_collapsed(weighted=True, prior='none', normalize_edge=False, remove_celebrities=False, dev=False, project_to_main_users=False, node_order='l2h', remove_mentions_with_degree_one=True):
    '''
    runs label propagation over the collapsed @-mention graph.
    In a collapsed @-mention graph, @-mentions which are not a member of target nodes (training,dev or test users) are
    removed from the graph. The target nodes which were previously connected through the removed @-mention nodes
    will be connected with an edge.
    If project_to_main_users is False the graph won't be collapsed and the complete @-mention graph will be used.
    '''
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    U_eval = []
    if dev:
        evalText = devText
        evalUsers = devUsers
        U_eval = U_dev
    else:
        evalText = testText
        evalUsers = testUsers
        U_eval = U_test

    U_all = U_train + U_eval
    assert len(U_all) == len(U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    
    mention_graph = nx.Graph()
    
    print('weighted=%s and prior=%s' % (weighted, prior))

    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    l = len(trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    mention_graph.add_nodes_from(node_id.values())
    for user, text in trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user_id] 
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if mention_graph.has_edge(user_id, mention):
                mention_graph[user_id][mention]['weight'] += freq
                # mention_graph[mention][user]['weight'] += freq/2.0
            else:
                mention_graph.add_edge(user_id, mention, weight=freq)
                # mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        user_id = node_id[user]
        mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user_id]
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if mention_graph.has_edge(user_id, mention):
                mention_graph[user_id][mention]['weight'] += freq
                # mention_graph[mention][user]['weight'] += freq/2.0
            else:
                mention_graph.add_edge(user_id, mention, weight=freq)
                # mention_graph.add_edge(mention, user, weight=freq/2.0)  
        
    
    trainuserid_location = {}
    evaluserid_location = {}
    trainLats = []
    trainLons = []
    node_location = {}
    dongle_nodes = []
    text_preds = {}
    for user, loc in trainUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)
        trainuserid_location[user_id] = (lat, lon)
        node_location[user_id] = (lat, lon)
        
    for user, loc in evalUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        evaluserid_location[user_id] = (lat, lon)
    
    print "the number of train nodes is " + str(len(trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    medianLat = np.median(trainLats)
    medianLon = np.median(trainLons)

    celebrity_threshold = celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = mention_graph.nodes_iter()
        for node in nodes:
            nbrs = mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in evaluserid_location and node not in trainuserid_location:
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
                mention_graph.remove_node(celebrity)
    
    dongle = False
    backoff = False
    text_direct = False
    if prior != 'none':
        if prior == 'backoff':
            backoff = True
        elif prior == 'prior':
            text_direct = True
        elif prior == 'dongle':
            dongle = True
        prior_file_path = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER - 1] + '-' + partitionMethod + '-' + str(BUCKET_SIZE) + '.pkl')
        print "reading prior text-based locations from " + prior_file_path
        if os.path.exists(prior_file_path):
            with open(prior_file_path, 'rb') as inf:
                preds, devPreds, U_test, U_dev, testProbs, devProbs = pickle.load(inf)
                if dev:
                    preds = devPreds
                    U_test = U_dev
                    testProbs = devProbs
                test_confidences = testProbs[np.arange(0, preds.shape[0]), preds]
                loss(preds=preds, U_test=U_test)
                if dongle:
                    mention_graph.add_nodes_from([u + '.dongle' for u in U_test])
                user_index = 0   
                for user, pred in zip(U_test, preds):
                    user_id = node_id[user]
                    lat = classLatMedian[str(pred)]
                    lon = classLonMedian[str(pred)]
                    if backoff:
                        text_preds[user_id] = (lat, lon)
                    if dongle:
                        dongle_node = user_id + '.dongle'
                        w = test_confidences[user_index]
                        mention_graph.add_edge(dongle_node, user_id, weight=w)
                        node_location[dongle_node] = (lat, lon)
                        dongle_nodes.append(dongle_node)
                    elif text_direct:
                        node_location[user_id] = (lat, lon)
                    user_index += 1
        else:
            print "prior file not found."




    if remove_mentions_with_degree_one:
        mention_nodes = set(mention_graph.nodes()) - set(node_id.values())
        mention_degree = mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = {node for node, degree in mention_degree.iteritems() if degree < 2}
        logging.info('found ' + str(len(one_degree_non_target)) + ' mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            mention_graph.remove_node(node)
            
    if project_to_main_users:
        logging.info('projecting the graph into the target user.')
        main_users = node_id.values()
        # mention_graph = bipartite.overlap_weighted_projected_graph(mention_graph, main_users, jaccard=False)
        mention_graph = collaboration_weighted_projected_graph(mention_graph, main_users, weight_str=None, degree_power=1, caller='lp')
        # mention_graph = collaboration_weighted_projected_graph(mention_graph, main_users, weight_str='weight')
        # mention_graph = bipartite.projected_graph(mention_graph, main_users)
    logging.info("Edge number: " + str(mention_graph.number_of_edges()))
    logging.info("Node number: " + str(mention_graph.number_of_nodes()))
    # results[str(project_to_main_users)] = mention_graph.degree().values()
    # return
    remove_betweeners = False
    if remove_betweeners:
        print("computing betweenness centrality of all nodes, takes a long time, sorry!")
        scores = nx.betweenness_centrality(mention_graph, weight='weight')
        i = 0
        percent_5 = len(scores) / 20 
        for w in sorted(scores, key=scores.get, reverse=True):
            i += 1
            if i < percent_5:
                mention_graph.remove_node(w)
    
    
        
    
    # check the effect of geolocation order on performance. The intuition is that
    # if the high confident nodes are geolocated first it may be better because noisy
    # predictions won't propagate from the first iterations.
    if node_order:
        node_degree = mention_graph.degree()
        if node_order == 'h2l':
            reverse_order = True
        else:
            reverse_order = False
        nodes = sorted(node_degree, key=node_degree.get, reverse=reverse_order)
        if node_order == 'random':
            random.shuffle(nodes)
        # nodes_unknown = [node for node in mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]
        nodes_unknown = [node for node in nodes if node not in trainuserid_location and node not in dongle_nodes]
    else:
        nodes_unknown = [node for node in mention_graph.nodes() if node not in trainuserid_location and node not in dongle_nodes]
    # find the cycles with 3 nodes and increase their edge weight
    increase_cyclic_edge_weights = False
    if increase_cyclic_edge_weights:
        increase_coefficient = 2
        cycls_3 = [c for c in list(nx.find_cliques(mention_graph)) if len(c) > 2]
        print(str(len(cycls_3)) + ' triangles in the graph.')
        # cycls_3 = [c for c in nx.cycle_basis(mention_graph) if len(c)==3]
        for c_3 in cycls_3:
            mention_graph[c_3[0]][c_3[1]]['weight'] *= increase_coefficient
            mention_graph[c_3[0]][c_3[2]]['weight'] *= increase_coefficient
            mention_graph[c_3[1]][c_3[2]]['weight'] *= increase_coefficient
        del cycls_3
    
    # remove (or decrease the weights of) the edges between training nodes which are very far from each other
    remove_inconsistent_edges = False
    if remove_inconsistent_edges:
        max_acceptable_distance = 161
        num_nodes_removed = 0
        edges = mention_graph.edges()
        edges = [(a, b) for a, b in edges if a in node_location and b in node_location]
        for node1, latlon1 in node_location.iteritems():
            for node2, latlon2 in node_location.iteritems():
                lat1, lon1 = latlon1
                lat2, lon2 = latlon2
                dd = distance(lat1, lon1, lat2, lon2)
                if dd > max_acceptable_distance:
                    if ((node1, node2) in edges or (node2, node1) in edges):
                        try:
                            mention_graph.remove_edge(node1, node2)
                            num_nodes_removed += 1
                        except:
                            pass
        print(str(num_nodes_removed) + ' edges removed from the graph') 
                
    use_shortest_paths = False
    shortest_paths = {}
    if use_shortest_paths:
        shortest_paths = nx.all_pairs_shortest_path_length(mention_graph, cutoff=3)
    
    
    converged = False
    logging.info("weighted " + str(weighted))
    max_iter = 5
    iter_num = 1
    logging.info("iterating with max_iter = " + str(max_iter))
    # if selfish = True, the nodes location would be added to that of its neighbours and then the median is computed. (it didn't improve the results on cmu)
    selfish = False
    while not converged:
        if node_order == 'random':
            random.shuffle(nodes_unknown)
        isolated_users = set()
        print "iter: " + str(iter_num)
        located_nodes_count = len(node_location)
        logging.info(str(located_nodes_count) + " nodes have location")
        for node in nodes_unknown:
            nbrs = mention_graph[node]
            nbrlats = []
            nbrlons = []
            nbr_edge_weights = []
            
            if selfish:
                if node in node_location:
                    self_lat, self_lon = node_location[node]
                    nbrlats.append(self_lat)
                    nbrlons.append(self_lon)
                
            for nbr in nbrs:
                if nbr in node_location:
                    lat, lon = node_location[nbr]
                    edge_weight = mention_graph[node][nbr]['weight']
                    nbrlats.append(lat)
                    nbrlons.append(lon)
                    if normalize_edge and weighted:
                        edge_weight_normalized = float(edge_weight) / (mention_graph.degree(nbr) * mention_graph.degree(node))
                        nbr_edge_weights.append(edge_weight_normalized)
                    # elif not weighted:
                    #    nbr_edge_weights.append(1)
                    else:
                        nbr_edge_weights.append(edge_weight)
                    
            if use_shortest_paths:
                if node in shortest_paths:
                    community_nbrs = shortest_paths[node]
                    for nbr, path_length in community_nbrs.iteritems():
                        if path_length > 1:
                            if nbr in node_location:
                                lat, lon = node_location[nbr]
                                nbrlats.append(lat)
                                nbrlons.append(lon)
            if len(nbrlons) > 0:
                nbr_median_lat, nbr_median_lon = weighted_median(nbrlats, nbr_edge_weights), weighted_median(nbrlons, nbr_edge_weights)
                node_location[node] = (nbr_median_lat, nbr_median_lon)

        iter_num += 1
        if iter_num == max_iter:
            converged = True
        
        if len(node_location) == located_nodes_count:
            logging.info("converged. No new nodes added in this iteration.")
            # converged = True
        distances = []
        isolated = 0
        for evaluser, loc in evalUsers.iteritems():
            lat, lon = locationStr2Float(loc)
            evaluserid = node_id[evaluser]
            if evaluserid not in node_location:
                isolated += 1
                isolated_users.add(evaluserid)
            if backoff and evaluserid in isolated_users:
                predicted_lat, predicted_lon = text_preds[evaluserid]
            else:
                predicted_lat, predicted_lon = node_location.get(evaluserid, (medianLat, medianLon))
            dist = distance(lat, lon, predicted_lat, predicted_lon)
            distances.append(dist)
        current_median = np.median(distances)
        current_mean = np.mean(distances)
        current_acc = 100 * len([d for d in distances if d < 161]) / float(len(distances))
        logging.info("mean: " + str(int(current_mean)))
        logging.info("median:" + str(int(current_median)))
        logging.info("Acc@161:" + str(current_acc))
        logging.info("isolated test users are " + str(isolated) + " out of " + str(len(distances)))
        
        compute_degree_error = False
        if compute_degree_error:
            degrees = []
            user_degree = mention_graph.degree([node_id[u] for u in evalUsers.keys()])
            for u in evalUsers:
                evalUserId = node_id[u]
                degree = user_degree[evalUserId]
                degrees.append(degree)
    
    return current_mean, current_median, current_acc

def LP_classification(weighted=True, prior='none', normalize_edge=False, remove_celebrities=False, dev=False, project_to_main_users=False, node_order='l2h', remove_mentions_with_degree_one=True):
    '''
    This function implements iterative label propagation as in Modified Adsorption without the regulariser term.
    This is not parallel and is slower than running Junto.
    The labels are the result of running k-d tree over the training coordinates which discretises
    the real-valued coordinates into several regions with different area but the same number of users.
    The label distribution of training users are kept unchanged. The label distribution of test/dev users
    are updated to the mean of their neighbours.
    If project_to_main_users is True the network will be collapsed (keeping just training and test/dev users) and
    if False, the complete @-mention graph will be used.
    '''
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    num_classes = len(categories)
    logging.info('running classification based label propagation') 
    U_eval = []
    if dev:
        evalText = devText
        evalUsers = devUsers
        U_eval = U_dev
        eval_classes = devClasses
    else:
        evalText = testText
        evalUsers = testUsers
        U_eval = U_test
        eval_classes = testClasses
    save_gr = False
    U_all = U_train + U_eval
    assert len(U_all) == len(U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))


    
    mention_graph = nx.Graph()
    graph_file_address = path.join(GEOTEXT_HOME, 'direct_graph.graphml')
    
    print('weighted=%s and prior=%s' % (weighted, prior))

    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    l = len(trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    mention_graph.add_nodes_from(node_id.values())
    for user, text in trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user_id] 
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if mention_graph.has_edge(user_id, mention):
                mention_graph[user_id][mention]['weight'] += freq
                # mention_graph[mention][user]['weight'] += freq/2.0
            else:
                mention_graph.add_edge(user_id, mention, weight=freq)
                # mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        user_id = node_id[user]
        mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        mentions = [m for m in mentions if m != user_id]
        mentionDic = Counter(mentions)
        mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if mention_graph.has_edge(user_id, mention):
                mention_graph[user_id][mention]['weight'] += freq
                # mention_graph[mention][user]['weight'] += freq/2.0
            else:
                mention_graph.add_edge(user_id, mention, weight=freq)
                # mention_graph.add_edge(mention, user, weight=freq/2.0)  
        
    
    trainuserid_location = {}
    evaluserid_location = {}
    trainLats = []
    trainLons = []
    node_location = {}
    dongle_nodes = []
    text_preds = {}
    
    for user, loc in trainUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)

        
    for user, loc in evalUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        evaluserid_location[user_id] = (lat, lon)
    
    print "the number of train nodes is " + str(len(trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    median_lat = np.median(trainLats)
    median_lon = np.median(trainLons)
    median_classIndx, dist = assignClass(median_lat, median_lon)
    celebrity_threshold = celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = mention_graph.nodes_iter()
        for node in nodes:
            nbrs = mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in node_id.values():
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
                mention_graph.remove_node(celebrity)
    





    if remove_mentions_with_degree_one:
        mention_nodes = set(mention_graph.nodes()) - set(node_id.values())
        mention_degree = mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = {node for node, degree in mention_degree.iteritems() if degree < 2}
        logging.info('found ' + str(len(one_degree_non_target)) + ' mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            mention_graph.remove_node(node)
            
    if project_to_main_users:
        logging.info('projecting the graph into the target user.')
        main_users = node_id.values()
        # mention_graph = bipartite.overlap_weighted_projected_graph(mention_graph, main_users, jaccard=False)
        mention_graph = collaboration_weighted_projected_graph(mention_graph, main_users, weight_str=None, degree_power=1, caller='lp')
        # mention_graph = collaboration_weighted_projected_graph(mention_graph, main_users, weight_str='weight')
        # mention_graph = bipartite.projected_graph(mention_graph, main_users)
    logging.info("Edge number: " + str(mention_graph.number_of_edges()))
    logging.info("Node number: " + str(mention_graph.number_of_nodes()))


    node_labeldist = {}
    logging.info('initialising user label distributions...')
    for node, id in node_id.iteritems():
        if id < len(U_train):
            label = trainClasses[node]
            dist = lil_matrix((1, len(categories)))
            dist[0, label] = 1
            node_labeldist[id] = dist
        
    
    # check the effect of geolocation order on performance. The intuition is that
    # if the high confident nodes are geolocated first it may be better because noisy
    # predictions won't propagate from the first iterations.
    if node_order:
        node_degree = mention_graph.degree()
        if node_order == 'h2l':
            reverse_order = True
        else:
            reverse_order = False
        nodes = sorted(node_degree, key=node_degree.get, reverse=reverse_order)
        if node_order == 'random':
            random.shuffle(nodes)
        # nodes_unknown = [node for node in mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]
        nodes_unknown = [node for node in nodes if node not in trainuserid_location and node not in dongle_nodes]
    else:
        nodes_unknown = [node for node in mention_graph.nodes() if node not in trainuserid_location and node not in dongle_nodes]

    converged = False
    logging.info("weighted " + str(weighted))
    max_iter = 5
    iter_num = 1
    logging.info("iterating with max_iter = " + str(max_iter))
    while not converged:
        if node_order == 'random':
            random.shuffle(nodes_unknown)
        isolated_users = set()
        print "iter: " + str(iter_num)
        located_nodes_count = len(node_labeldist)
        logging.info(str(located_nodes_count) + " nodes have location")
        for node in nodes_unknown:
            should_be_updated = False
            neighbors_labeldist = lil_matrix((1, len(categories)))
            nbrs = mention_graph[node]    
            for nbr, edge_data in nbrs.iteritems():
                if nbr in node_labeldist:
                    should_be_updated = True
                    nbrlabeldist = node_labeldist[nbr]
                    edge_weight = edge_data['weight']
                    neighbors_labeldist = neighbors_labeldist + edge_weight * nbrlabeldist
            if should_be_updated:
                old_labeldist = node_labeldist.get(node, csr_matrix((1, len(categories))))
                new_labeldist = old_labeldist + neighbors_labeldist
                # inplace normalization
                normalize(new_labeldist, norm='l1', copy=False)
                node_labeldist[node] = new_labeldist
        eval_predictions = []
        eval_confidences = []
        for u in U_eval:
            u_id = node_id[u]
            prediction = -1
            confidence = 0
            if u_id in node_labeldist:
                labeldist = node_labeldist[u_id].toarray()
                prediction = np.argmax(labeldist)
                confidence = np.max(labeldist) 
            else:
                isolated_users.add(u_id)
                prediction = int(median_classIndx)
            eval_predictions.append(prediction)
        loss(preds=eval_predictions, U_test=U_eval, save_results_bucket_size=True, results_key=('classification', project_to_main_users, node_order)) 
        iter_num += 1
        if iter_num == max_iter:
            converged = True
        
        if len(node_labeldist) == located_nodes_count:
            logging.info("converged. No new nodes added in this iteration.")
            # converged = True
        
    
    



def collaboration_weighted_projected_graph(B, nodes, weight_str=None, degree_power=1, caller='lp'):
    r"""Newman's weighted projection of B onto one of its node sets.

    The collaboration weighted projection is the projection of the
    bipartite network B onto the specified nodes with weights assigned
    using Newman's collaboration model [1]_:

    .. math::
        
        w_{v,u} = \sum_k \frac{\delta_{v}^{w} \delta_{w}^{k}}{k_w - 1}

    where `v` and `u` are nodes from the same bipartite node set,
    and `w` is a node of the opposite node set. 
    The value `k_w` is the degree of node `w` in the bipartite
    network and `\delta_{v}^{w}` is 1 if node `v` is
    linked to node `w` in the original bipartite graph or 0 otherwise.
 
    The nodes retain their attributes and are connected in the resulting
    graph if have an edge to a common node in the original bipartite
    graph.

    Parameters
    ----------
    B : NetworkX graph 
      The input graph should be bipartite. 

    nodes : list or iterable
      Nodes to project onto (the "bottom" nodes).

    Returns
    -------
    Graph : NetworkX graph 
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(5)
    >>> B.add_edge(1,5)
    >>> G = bipartite.collaboration_weighted_projected_graph(B, [0, 2, 4, 5])
    >>> print(G.nodes())
    [0, 2, 4, 5]
    >>> for edge in G.edges(data=True): print(edge)
    ... 
    (0, 2, {'weight': 0.5})
    (0, 5, {'weight': 0.5})
    (2, 4, {'weight': 1.0})
    (2, 5, {'weight': 0.5})
    
    Notes
    ------
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See Also
    --------
    is_bipartite, 
    is_bipartite_node_set, 
    sets, 
    weighted_projected_graph,
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph,
    projected_graph 

    References
    ----------
    .. [1] Scientific collaboration networks: II. 
        Shortest paths, weighted networks, and centrality, 
        M. E. J. Newman, Phys. Rev. E 64, 016132 (2001).
    """
    if B.is_multigraph():
        raise nx.NetworkXError("not defined for multigraphs")
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.node[n]) for n in nodes)
    direct_edge_counter = 0
    for v1, v2 in B.edges_iter(data=False):
        if type(v1) == int and type(v2) == int:
            w = (1.0 / len(B[v1]) + 1.0 / len(B[v2]))
            G.add_edge(v1, v2, weight=w) 
            direct_edge_counter += 1
    logging.info('direct edge count: ' + str(direct_edge_counter))
    i = 0
    tenpercent = len(nodes) / 10
    for n1 in nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        unbrs = set(B[n1])
        nbrs2 = set((n for nbr in unbrs for n in B[nbr])) - set([n1])
        nbrs2 = [n for n in nbrs2 if type(n) == int]
            # pass
        for n2 in nbrs2:
            weight = 0
            if G.has_edge(n1, n2):
                weight += G[n1][n2]['weight'] 
            vnbrs = set(pred[n2])
            common = unbrs & vnbrs
            del vnbrs

            if weight_str is not None:
                weight += sum([1.0 * B[n1][n][weight_str] * B[n2][n][weight_str] / ((len(B[n]) - 1) ** degree_power) for n in common if len(B[n]) > 1])
            else:
                    weight += sum([1.0 / ((len(B[n]) - 1) ** degree_power) for n in common if len(B[n]) > 1])
            if weight != 0:
                G.add_edge(n1, n2, weight=weight)
    
    
    return G



def junto_postprocessing(multiple=False, dev=False, method='median', celeb_threshold=5, weighted=False, text_prior='none', postfix=''):
    EVALUATE_REAL_VALUED = False

    lats = classLatMedian.values()
    lons = classLonMedian.values()
    median_lat = np.median(lats)
    median_lon = np.median(lons)
    classIndx, dist = assignClass(median_lat, median_lon)
    trainUsersLower = [u.lower() for u in sorted(trainUsers)]
    
    
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    if dev:
        U_eval = U_dev
        devStr = '.dev'
    else:
        U_eval = U_test
        devStr = ''
    result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER - 1] + '-' + method + '-' + str(BUCKET_SIZE) + '.pkl')
    if text_prior != 'none':
        logging.info("reading (preds, devPreds, U_test, U_dev, testProbs, devProbs) from " + result_dump_file)
        with open(result_dump_file, 'rb') as inf:
            preds_text, devPreds_text, U_test_text, U_dev_text, testProbs_text, devProbs_text = pickle.load(inf)
            logging.info("text test results:")
            loss(preds_text, U_test_text)
            logging.info("text dev results:")
            loss(devPreds_text, U_dev_text)
            
            if dev:
                text_preds = devPreds_text
                text_probs = devProbs_text
                assert U_dev == U_dev_text, "text users are not equal to loaded dev users"
                
            else:
                text_preds = preds_text
                text_probs = testProbs_text
                assert U_test == U_test_text, "text users are not equal to loaded test users"
    
    # split a training set and a test set
    Y_test = np.asarray([testClasses[u] for u in U_test])
    Y_dev = np.asarray([devClasses[u] for u in U_dev])
    
    textStr = '.' + text_prior
    if text_prior == 'none':
        textStr = ''
    weightedStr = '.weighted'
    if not weighted:
        weightedStr = ''
    

        
    if not multiple:
        
        files = [path.join(GEOTEXT_HOME, 'label_prop_output_' + DATASETS[DATASET_NUMBER - 1] + '_' + method + '_' + str(BUCKET_SIZE) + '_' + str(celeb_threshold) + devStr + textStr + weightedStr + postfix)]
    else:
        junto_output_dir = '/home/arahimi/git/junto-master/examples/simple/data/outputs_' + DATASETS[DATASET_NUMBER - 1]
        files = glob.glob(junto_output_dir + '/label_prop_output*')
        files = sorted(files)
    # feature_extractor2(min_df=50)

    for junto_output_file in files:   
        id_name_file = path.join(GEOTEXT_HOME, 'id_user' + '_' + method + '_' + str(BUCKET_SIZE) + devStr + textStr + weightedStr) 
        logging.info("output file: " + junto_output_file)
        logging.info("id_name file: " + id_name_file)
        name_id = {}
        id_pred = {}
        name_pred = {}
        id_name = {}
        with codecs.open(id_name_file, 'r', 'utf-8') as inf:
            for line in inf:
                fields = line.split()
                name_id[fields[1]] = fields[0]
                id_name[fields[0]] = fields[1]
        

        dummy_count = 0
        with codecs.open(junto_output_file, 'r', 'utf-8') as inf:  
            print junto_output_file               
            # real valued results were not good
            if EVALUATE_REAL_VALUED:
                distances = []
                for line in inf:
                    fields = line.split()
                    if len(fields) == 11:
                        uid = fields[0]
                        if '.T' in uid:
                            continue
                        lat = float(fields[4])
                        lon = float(fields[8])
                        u = U_test[int(uid) - len(trainUsers)]
                        
                        lat2, lon2 = locationStr2Float(userLocation[u])
                        distances.append(distance(lat, lon, lat2, lon2))
                print "results"
                print str(np.mean(distances))
                print str(np.median(distances))
            else:
                nopred = []
                for line in inf:
                    fields = line.split('\t')
                    uid = fields[0]
                    if '.T' in uid:
                        continue
                    u = id_name[uid]
                    test_user_inconsistency = 0
                    second_option_selected = 0
                    if u in U_eval:
                        user_index = U_eval.index(u)
                        try:
                            label_scores = fields[-3]
                            label = label_scores.split()[0]
                            labelProb = float16(label_scores.split()[1])
                            if label == '__DUMMY__':
                                # pass
                                logging.info('choosing second ranked label as the first one is a dummy!')
                                label = label_scores.split()[2]
                                labelProb = float16(label_scores.split()[2])
                                second_option_selected += 1
                        except:
                            print fields
                        
                        # Tracer()()
                        if label == '__DUMMY__':
                            dummy_count += 1
                            if text_prior == 'backoff':
                                text_predition = str(text_preds[user_index])
                                label = text_predition
                            else:
                                # label = str(len(categories) / 2)
                                label = str(classIndx)
                                
                        id_pred[fields[0]] = label
                    else:
                        if u not in trainUsersLower:
                            nopred.append((uid, u))
                if len(nopred) > 0:
                    print "no predition for these nodes:" + str(nopred)
                    logging.info("no prediction for the above nodes.")
                    sys.exit()
                logging.info("users with second predicted label: " + str(second_option_selected))
                

                preds = []
                user_not_in_network = 0
                doubles_found = 0
                for u in U_eval:
                    if u.lower() not in name_id:
                        ud = u + '_double00'
                        doubles_found += 1
                        uid = name_id[ud.lower()]
                    else:
                        uid = name_id[u.lower()]
                    if uid in id_pred:
                        pred = id_pred[uid]
                        name_pred[u] = pred
                    else:
                        # print 'user %d not in network predictions.'
                        user_not_in_network += 1
                        if text_prior == 'backoff':
                            text_predition = str(text_preds[int(uid) - len(trainUsersLower)])
                            pred = text_predition
                        else:
                            # pred = str(len(categories) / 2)  
                            pred = classIndx
                        name_pred[u] = pred

                    preds.append(int(pred))
                
                print "doubles found is " + str(doubles_found)
                print "users with dummy labels: " + str(dummy_count)
                print "users not in network: " + str(user_not_in_network)
                print "total number of users: " + str(len(U_eval))
                # print preds
                # print [int(i) for i in Y_test.tolist()]    
                mad_mean, mad_median, mad_acc = loss(preds, U_eval, save_results_bucket_size=True, results_key=None)
    return mad_mean, mad_median, mad_acc
                
                    

def weighted_median(values, weights):
    ''' compute the weighted median of values list. The 
    weighted median is computed as follows:
    1- sort both lists (values and weights) based on values.
    2- select the 0.5 point from the weights and return the corresponding values as results
    e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are probabilities.
    sorted values = [0, 1, 3] and corresponding sorted weights = [0.6, 0.1, 0.3] the 0.5 point on
    weight corresponds to the first item which is 0. so the weighted median is 0.'''
    
    # convert the weights into probabilities
    sum_weights = sum(weights)
    weights = np.array([(w * 1.0) / sum_weights for w in weights])
    # sort values and weights based on values
    values = np.array(values)
    sorted_indices = np.argsort(values)
    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]
    # select the median point
    it = np.nditer(weights_sorted, flags=['f_index'])
    accumulative_probability = 0
    median_index = -1
    while not it.finished:
        accumulative_probability += it[0]
        if accumulative_probability >= 0.5:
            median_index = it.index
            return values_sorted[median_index]
        elif accumulative_probability == 0.5:
            median_index = it.index
            it.iternext()
            next_median_index = it.index
            return mean(values_sorted[[median_index, next_median_index]])
        it.iternext()

    return values_sorted[median_index]


initialize(partitionMethod=partitionMethod, granularity=BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=False)
if 'text-classification' in models_to_run:
    t_mean, t_median, t_acc, d_mean, d_median, d_acc = asclassification(granularity=BUCKET_SIZE, partitionMethod=partitionMethod, use_mention_dictionary=False, binary=binary, sublinear=sublinear, penalty=penalty, fit_intercept=fit_intercept, norm=norm, use_idf=use_idf)
if 'network-lp-regression-collapsed' in models_to_run:
    LP_collapsed(weighted=False, prior='none', normalize_edge=False, remove_celebrities=True, dev=True, project_to_main_users=True, node_order='random', remove_mentions_with_degree_one=True)
if 'network-lp-regression' in models_to_run:
    LP(weighted=False, prior='none', normalize_edge=False, remove_celebrities=True, dev=True, node_order='random')
if 'network-lp-classification' in models_to_run:
    LP_classification(weighted=True, prior='none', normalize_edge=False, remove_celebrities=True, dev=True, project_to_main_users=True, node_order='random', remove_mentions_with_degree_one=True)



# junto_postprocessing(multiple=False, dev=False, text_confidence=1.0, method=partitionMethod, celeb_threshold=5, weighted=True, text_prior=True)

