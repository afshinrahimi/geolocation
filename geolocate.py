'''
Created on 26 Feb 2016

@author: af
'''

from IPython.core.debugger import Tracer
import codecs
from collections import defaultdict, Counter
from datetime import datetime
import glob
import gzip
from haversine import haversine
import logging
import os
import pdb
import pickle
import random
import re
from scipy import mean
from scipy.sparse import csr_matrix
from scipy.sparse.lil import lil_matrix
from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.utils.extmath import density
import sys
import time
import scipy as sp
import networkx as nx
import numpy as np
import params
import utils
import scipy.sparse as sparse
#import embedding_optimiser as edgexplain
from sklearn.externals import joblib
from os import path
from params import classLatMedian
import operator
from sklearn.linear_model.logistic import LogisticRegression

random.seed(0)
__docformat__ = 'restructedtext en'
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


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


def users(file, type='train', write=False, readText=True):
    if readText:
        logging.info("Text is being read.")
        if params.TEXT_ONLY:
            logging.info('mentions, hashtags and urls are removed.')
    
    # with codecs.open(file, 'r', encoding=params.data_encoding) as inf:
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
            if params.TEXT_ONLY and readText:
                text = re.sub(r"(?:\@|#|https?\://)\S+", "", text)
            locStr = lat + ',' + lon
            locFloat = (float(lat), float(lon))
            params.userLocation[user] = locStr
            if type == 'train':
                params.trainUsers[user] = locStr
                if readText:
                    params.trainText[user] = text
                users_in_loc = params.locationUser.get(locFloat, [])
                users_in_loc.append(user)
                params.locationUser[locFloat] = users_in_loc
            elif type == 'test':
                params.testUsers[user] = locStr
                if readText:
                    params.testText[user] = text
            elif type == 'dev':
                params.devUsers[user] = locStr
                if readText:
                    params.devText[user] = text

        








def discretize_kdtree(granularity, partitionMethod):
    filename = path.join(params.GEOTEXT_HOME, 'processed_data/' + str(granularity).strip() + '_' + partitionMethod + '_clustered.train')
    print "reading " + filename
    params.U_train = sorted(params.trainUsers)
    params.U_test = sorted(params.testUsers)
    params.U_dev = sorted(params.devUsers)
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    with codecs.open(filename, 'r', encoding=params.data_encoding) as inf:
        for line in inf:
            points = []
            minlat = 1000
            maxlat = -1000
            minlon = 1000
            maxlon = -1000
            fields = set(line.strip().split('\t'))
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
        label = str(i).strip()
        params.categories.append(label)
        params.classLatMedian[label] = medianlat
        params.classLonMedian[label] = medianlon
  
        
        for location in cluster:
            locusers = params.locationUser[(location[0], location[1])]
            user_class = dict(zip(locusers, [i] * len(locusers)))
            params.trainClasses.update(user_class)
    logging.info("train directories created and class median and mean lat,lon computed. params.trainfile: " + filename)
    devDistances = []
    for user in params.devUsers:
        locationStr = params.devUsers[user]
        latlon = locationStr.split(',')
        latitude = float(latlon[0])
        longitude = float(latlon[1])
        classIndex, dist = assignClass(latitude, longitude)
        devDistances.append(dist)
        params.devClasses[user] = int(classIndex)
    
    testDistances = []
    for user in params.testUsers:
        locationStr = params.testUsers[user]
        latlon = locationStr.split(',')
        latitude = float(latlon[0])
        longitude = float(latlon[1])
        classIndex, dist = assignClass(latitude, longitude)
        testDistances.append(dist)
        params.testClasses[user] = int(classIndex)

            


        
    Report_Ideal = False
    if Report_Ideal:
        print "Ideal mean dev distance is " + str(np.mean(devDistances))
        print "Ideal median dev distance is " + str(np.median(devDistances))
        print "Ideal Acc@161 dev is " + str(len([dist for dist in devDistances if dist < 161]) / (len(devDistances) + 0.0))
        
        print "Ideal mean test distance is " + str(np.mean(testDistances))
        print "Ideal median test distance is " + str(np.median(testDistances))
        print "Ideal Acc@161 test is " + str(len([dist for dist in testDistances if dist < 161]) / (len(testDistances) + 0.0))

def descritize_users(n_clusters=100):
    params.U_train = sorted(params.trainUsers)
    params.U_test = sorted(params.testUsers)
    params.U_dev = sorted(params.devUsers)
    from sklearn.cluster import KMeans
    points = np.array([locationStr2Float(params.trainUsers[u]) for u in params.U_train])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit_transform(points)
    sample_clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    #with open(path.join(params.GEOTEXT_HOME, str(n_clusters) + '-kmeans.pkl'), 'wb') as outf:
    #    pickle.dump((sample_clusters, cluster_centers), outf)
    params.classLatMedian.clear()
    params.classLonMedian.clear()
    params.categories = []
    params.trainClasses.clear()
    params.devClasses.clear()
    params.testClasses.clear()
    for i in range(cluster_centers.shape[0]):
        lat, lon = cluster_centers[i, 0], cluster_centers[i, 1]
        params.classLatMedian[str(i)] = lat
        params.classLonMedian[str(i)] = lon
        params.categories.append(str(i))

    for i, u in enumerate(params.U_train):
        params.trainClasses[u] = sample_clusters[i]
    for u in params.U_test:
        lat, lon = locationStr2Float(params.testUsers[u]) 
        params.testClasses[u], dist = assignClass(lat, lon)
    for u in params.U_dev:
        lat, lon = locationStr2Float(params.devUsers[u])
        params.devClasses[u], dist = assignClass(lat, lon)

    
def descritize_points(points, n_clusters=100):    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit_transform(points)
    sample_clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    params.categories = []
    params.trainClasses.clear()
    params.devClasses.clear()
    params.testClasses.clear()
    for i in range(cluster_centers.shape[0]):
        lat, lon = cluster_centers[i, 0], cluster_centers[i, 1]
        params.classLatMedian[str(i)] = lat
        params.classLonMedian[str(i)] = lon
        params.categories.append(str(i))
    for i, u in enumerate(params.U_train):
        params.trainClasses[u] = sample_clusters[i]
    for u in params.U_test:
        lat, lon = locationStr2Float(params.testUsers[u]) 
        params.testClasses[u] = assignClass(lat, lon)
    for u in params.U_dev:
        lat, lon = locationStr2Float(params.devUsers[u])
        params.devClasses[u] = assignClass(lat, lon)
    
    plot = True
    if plot:
        from diagrams import plot_regions
        cluster_points = defaultdict(list)
        for i, u in enumerate(params.U_train):
            cluster_points[sample_clusters[i]].append(locationStr2Float(params.trainUsers[u]))
        plot_regions(cluster_points, filename='plot'+ str(params.i) + '.jpg')
        params.i += 1
            
        
    
def assignClass(latitude, longitude):
    '''
    Given a coordinate find the class whose median is the closest point. Then return the index of that class.
    This function can be used for parameter tuning with validation data and evaluation with test data.
    '''
    minDistance = 1000000
    classIndex = -1
    for i in params.classLatMedian:
        lat = params.classLatMedian[str(i).strip()]
        lon = params.classLonMedian[str(i).strip()]
        dist = distance(latitude, longitude, lat, lon)
        if dist < minDistance:
            minDistance = dist
            classIndex = i
        if int(dist) == 0:
            return classIndex, minDistance
    return classIndex, minDistance




    

def size_mb(docs):
    return sum(len(s.encode(encoding=params.data_encoding)) for s in docs) / 1e6

def evaluate(preds, U_test, categories, scores):
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    confidences = []
    randomConfidences = []
    gmm = False
    for i in range(0, len(preds)):
        user = params.U_test[i]
        location = params.userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        # gaussian mixture model
        if gmm:
            sumMedianLat = 0
            sumMedianLon = 0
            numClasses = len(params.categories)
            sortedScores = sorted(scores[i], reverse=True)
            top1Score = sortedScores[0]
            top2Score = sortedScores[1]
            print top1Score
            print top2Score
            for c in range(0, numClasses):
                score = scores[i][c]
                category = params.categories[c]
                medianlat = params.classLatMedian[category]  
                medianlon = params.classLonMedian[category]  
                sumMedianLat += score * medianlat
                sumMedianLon += score * medianlon
            distances.append(distance(lat, lon, sumMedianLat, sumMedianLon)) 
            
        else:
            prediction = params.categories[preds[i]]
            if scores != None:
                confidence = scores[i][preds[i]] 
                confidences.append(confidence)
            medianlat = params.classLatMedian[prediction]  
            medianlon = params.classLonMedian[prediction]    
            distances.append(distance(lat, lon, medianlat, medianlon))
            sumMedianDistance = sumMedianDistance + distance(lat, lon, medianlat, medianlon)

    # averageMeanDistance = sumMeanDistance / float(len(preds))
    # averageMedianDistance = sumMedianDistance / float(len(preds))
    # print "Average mean distance is " + str(averageMeanDistance)
    # print "Average median distance is " + str(averageMedianDistance)
    print "Mean distance is " + str(np.mean(distances))
    print "Median distance is " + str(np.median(distances))



            
def error(predicted_label, user):
    lat1, lon1 = locationStr2Float(params.userLocation[user])
    lat2 = params.classLatMedian[predicted_label]  
    lon2 = params.classLonMedian[predicted_label]
    return distance(lat1, lon1, lat2, lon2)         

def loss(preds, U_eval, save_results=False, verbose=True, error_analysis=False):
    if len(preds) != len(U_eval): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_eval))
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
    for i in range(0, len(preds)):
        user = U_eval[i]
        location = params.userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        user_original_class, minDistance = assignClass(lat, lon)
        distances_from_center.append(distance(lat, lon, center_of_us[0], center_of_us[1]))
        distances_from_nyc.append(distance(lat, lon, nyc[0], nyc[1]))
        distances_from_la.append(distance(lat, lon, la[0], la[1]))

        prediction = params.categories[preds[i]]
        medianlat = params.classLatMedian[prediction]  
        medianlon = params.classLonMedian[prediction]  
        user_location[user] = (medianlat, medianlon)    
        dd = distance(lat, lon, medianlat, medianlon)
        distances.append(dd)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    if verbose:
        logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))

    extra_info = False
    if extra_info:
        print "Mean distance from center of us is " + str(int(np.mean(distances_from_center)))
        print "Median distance from center of us is " + str(int(np.median(distances_from_center)))
        print "Mean distance from nyc is " + str(int(np.mean(distances_from_nyc)))
        print "Median distance from nyc is " + str(int(np.median(distances_from_nyc)))
        print "Mean distance from la is " + str(int(np.mean(distances_from_la)))
        print "Median distance from la is " + str(int(np.median(distances_from_la)))
    if U_eval[0] == params.U_dev[0] and save_results:
        params.dev_acc.append(acc_at_161)
        params.dev_mean.append(np.mean(distances))
        params.dev_median.append(np.median(distances))
    
    if error_analysis:
        logging.info('error analysis, dumping ...')
        min_error_distance_for_anaysis = sorted(distances)[-1000]
        with codecs.open('error_analysis.txt', 'w', encoding=params.data_encoding) as fout:
            for i in range(len(preds)):
                dd = distances[i]
                if dd < min_error_distance_for_anaysis:
                    continue
                u = U_eval[i]
                real_location = params.userLocation[u]
                predicted_location = user_location[u]
                if U_eval[0] == params.U_dev[0]:
                    text = params.devText[u]
                elif U_eval[0] == params.U_test[0]:
                    text = params.testText[u]
                line = [u, str(dd), real_location, str(predicted_location[0])+','+str(predicted_location[1]), text]
                fout.write(('\t'.join(line)+'\n').decode(params.data_encoding))
        
    return np.mean(distances), np.median(distances), acc_at_161

def train_regression_models():
    from sklearn.neighbors import NearestNeighbors
    class_users = defaultdict(list)
    user_index = {u:i for i,u in enumerate(params.U_train)}
    class_knn = {}
    for u, c in params.trainClasses.iteritems():
        class_users[c].append(user_index[u])
    for c in params.categories:
        #get the important dimensions from text classification model
        
        important_dimensions = params.clf.coef_[int(c)].nonzero()[1]
        X_region = params.X_train[class_users[int(c)]]
        X_region = X_region[:, important_dimensions]
        knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
        knn.fit(X_region)
        class_knn[int(c)] = (knn, class_users[int(c)], important_dimensions)
    return class_knn    

def train_classification_models():
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.cluster import KMeans
    import kdtree
    do_kmeans = True
    class_users = defaultdict(list)
    user_index = {u:i for i,u in enumerate(params.U_train)}
    index_user = {i:u for u,i in user_index.iteritems()}
    class_clf = {}
    for u, c in params.trainClasses.iteritems():
        class_users[c].append(user_index[u])
    for c in params.categories:
        #get the important dimensions from text classification model
        users = class_users[int(c)]
        points = [locationStr2Float(params.trainUsers[index_user[i]]) for i in users]
        if do_kmeans:
            kmeans = KMeans(n_clusters=min(4, len(points)), random_state=0)
            kmeans.fit(np.array(points))
            cluster_centers = kmeans.cluster_centers_
            Y_region = kmeans.labels_
        else:
            clusterer = kdtree.KDTreeClustering(bucket_size=min(params.BUCKET_SIZE / 4, len(points)))
            clusterer.fit(points)
            Y_region = clusterer.get_clusters()
            #print clusterer.num_clusters
            
        user_region = {index_user[uid]: Y_region[i] for i, uid in enumerate(users)}
        region_lats = defaultdict(list)
        region_lons = defaultdict(list)
        for u, r in user_region.iteritems():
            lat, lon = locationStr2Float(params.trainUsers[u])
            region_lats[r].append(lat)
            region_lons[r].append(lon)
        region_median_lat = {r:np.median(lats) for r, lats in region_lats.iteritems()}
        region_median_lon = {r:np.median(lons) for r, lons in region_lons.iteritems()}
        region_median = {r: (region_median_lat[r], region_median_lon[r]) for r in region_lats}
        
        #important_dimensions = params.clf.coef_[int(c)].nonzero()[1]
        X_region = params.X_train[users]
        #X_region = X_region[:, important_dimensions]
        #clf = MultinomialNB()
        clf = SGDClassifier(loss='log', alpha=1e-4, penalty='elasticnet', l1_ratio=0.9, learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=10, fit_intercept=True)
        clf.fit(X_region, Y_region)

        class_clf[int(c)] = (clf, users, region_median)
    return class_clf 

def train_clustering_models():
    from sklearn import mixture
    class_users = defaultdict(list)
    user_index = {u:i for i,u in enumerate(params.U_train)}
    index_user = {i:u for u,i in user_index.iteritems()}    
    for u, c in params.trainClasses.iteritems():
        class_users[c].append(user_index[u])
    class_clf = {}
    for c in params.categories:
        c = int(c)
        users = class_users[c]
        X_region = params.X_train[users]
        important_dimensions = params.clf.coef_[c].nonzero()[1]
        #important_dimensions = np.array(range(params.clf.coef_[c].shape[1]))
        X_region = X_region[:, important_dimensions]
        clusterer = mixture.GMM(n_components=4)
        Y_region = clusterer.fit_predict(X_region.toarray())
        user_region = {index_user[uid]: Y_region[i] for i, uid in enumerate(users)}
        region_lats = defaultdict(list)
        region_lons = defaultdict(list)
        for u, r in user_region.iteritems():
            lat, lon = locationStr2Float(params.trainUsers[u])
            region_lats[r].append(lat)
            region_lons[r].append(lon)
        region_median_lat = {r:np.median(lats) for r, lats in region_lats.iteritems()}
        region_median_lon = {r:np.median(lons) for r, lons in region_lons.iteritems()}
        region_median = {r: (region_median_lat[r], region_median_lon[r]) for r in region_lats}
        class_clf[c] = (clusterer, users, region_median, important_dimensions)   
    return class_clf     
        
def loss_clustering(preds, U_eval):
    
    if len(preds) != len(U_eval): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_eval))
        print "fatal error!"
        sys.exit()
    #check if it is dev or test
    if U_eval[0] in params.U_dev:
        dev = True
    else:
        dev = False
    def closest_neighbour_classes():
        neighbours = defaultdict(dict)
        sorted_neighbours = {}
        for c1 in params.classLatMedian:
            for c2 in params.classLatMedian:
                if c1 == c2:
                    continue
                lat1, lon1 = params.classLatMedian[c1], params.classLonMedian[c1]
                lat2, lon2 = params.classLatMedian[c2], params.classLonMedian[c2]
                dd = distance(lat1, lon1, lat2, lon2)
                neighbours[c1][c2] = dd
        for c, nbrs in neighbours.iteritems():
            sorted_nbrs = sorted(nbrs.items(), key=operator.itemgetter(1))
            sorted_neighbours[c] = sorted_nbrs
        return sorted_neighbours
                
    #sorted_neighbours = closest_neighbour_classes()    
    class_clf = train_clustering_models()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    inclass_closest_point_distances = []
    global_closest_point_distances = []
    inclass_gmm_distances = []
    user_location = {}
    acc = 0.0
    center_of_us = (39.50, -98.35)
    nyc = (40.7127, -74.0059)
    la = (34.0500, -118.2500)
    distances_from_nyc = []
    distances_from_la = []
    distances_from_center = []
    #global_closest = False
    for i in range(0, len(preds)):
        user = U_eval[i]
        location = params.userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        user_original_class, minDistance = assignClass(lat, lon)

        prediction = params.categories[preds[i]]
        medianlat = params.classLatMedian[prediction]  
        medianlon = params.classLonMedian[prediction]
        #nearby_classes =  [c[0] for c in sorted_neighbours[prediction][0:1]]
        #nearby_classes = []
        #nearby_classes.append(prediction)
        #nearby_classes = set(nearby_classes)
        
        #training_points_in_class = {u:locationStr2Float(params.trainUsers[u]) for u, c in params.trainClasses.iteritems() if str(c) in nearby_classes or global_closest}
        #u_point_distances = {u:distance(lat, lon, l[0], l[1]) for u, l in training_points_in_class.iteritems()}
        #closest_point =  min(u_point_distances, key=u_point_distances.get)
        #inclass_closest_distance = u_point_distances[closest_point]
        #inclass_closest_point_distances.append(inclass_closest_distance)
        
        #regress within predicted class using knn
        clf, users, region_median, important_dimensions = class_clf[preds[i]]
        if dev:
            X_u = params.X_dev[i, important_dimensions]
        else:
            X_u = params.X_test[i, important_dimensions]

        region = clf.predict(X_u.toarray())[0]
        nblat, nblon = region_median[region]
        inclass_gmm_distances.append(distance(lat, lon, nblat, nblon))   

        distances.append(distance(lat, lon, medianlat, medianlon))

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    print "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161))
    print
    #inclass_acc_at_161 = 100 * len([d for d in inclass_closest_point_distances if d < 161]) / float(len(inclass_closest_point_distances))
    #print "In Class closest -- Mean: " + str(int(np.mean(inclass_closest_point_distances))) + " Median: " + str(int(np.median(inclass_closest_point_distances))) + " Acc@161: " + str(int(inclass_acc_at_161))
    #print
    nn_acc_at_161 = 100 * len([d for d in inclass_gmm_distances if d < 161]) / float(len(inclass_gmm_distances))
    print "In Class classification -- Mean: " + str(int(np.mean(inclass_gmm_distances))) + " Median: " + str(int(np.median(inclass_gmm_distances))) + " Acc@161: " + str(int(nn_acc_at_161))
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
def loss_regression(preds, U_eval):
    
    if len(preds) != len(U_eval): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_eval))
        print "fatal error!"
        sys.exit()
    #check if it is dev or test
    if U_eval[0] in params.U_dev:
        dev = True
    else:
        dev = False
    def closest_neighbour_classes():
        neighbours = defaultdict(dict)
        sorted_neighbours = {}
        for c1 in params.classLatMedian:
            for c2 in params.classLatMedian:
                if c1 == c2:
                    continue
                lat1, lon1 = params.classLatMedian[c1], params.classLonMedian[c1]
                lat2, lon2 = params.classLatMedian[c2], params.classLonMedian[c2]
                dd = distance(lat1, lon1, lat2, lon2)
                neighbours[c1][c2] = dd
        for c, nbrs in neighbours.iteritems():
            sorted_nbrs = sorted(nbrs.items(), key=operator.itemgetter(1))
            sorted_neighbours[c] = sorted_nbrs
        return sorted_neighbours
                
    sorted_neighbours = closest_neighbour_classes()    
    class_knn = train_regression_models()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    inclass_closest_point_distances = []
    global_closest_point_distances = []
    inclass_nearestn_distances = []
    user_location = {}
    acc = 0.0
    center_of_us = (39.50, -98.35)
    nyc = (40.7127, -74.0059)
    la = (34.0500, -118.2500)
    distances_from_nyc = []
    distances_from_la = []
    distances_from_center = []
    global_closest = False
    for i in range(0, len(preds)):
        user = U_eval[i]
        location = params.userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        user_original_class, minDistance = assignClass(lat, lon)

        prediction = params.categories[preds[i]]
        medianlat = params.classLatMedian[prediction]  
        medianlon = params.classLonMedian[prediction]
        nearby_classes =  [c[0] for c in sorted_neighbours[prediction][0:1]]
        nearby_classes = []
        nearby_classes.append(prediction)
        nearby_classes = set(nearby_classes)
        
        training_points_in_class = {u:locationStr2Float(params.trainUsers[u]) for u, c in params.trainClasses.iteritems() if str(c) in nearby_classes or global_closest}
        u_point_distances = {u:distance(lat, lon, l[0], l[1]) for u, l in training_points_in_class.iteritems()}
        closest_point =  min(u_point_distances, key=u_point_distances.get)
        inclass_closest_distance = u_point_distances[closest_point]
        inclass_closest_point_distances.append(inclass_closest_distance)
        
        #regress within predicted class using knn
        knn, uids, important_features = class_knn[preds[i]]
        if dev:
            X_u = params.X_dev[i][:, important_features]
        else:
            X_u = params.X_test[i][:, important_features]

        nearest_neighbor_indices = knn.kneighbors(X_u)[1][0].tolist()
        nearest_neighbors = [uids[_id] for _id in nearest_neighbor_indices]
        coordinates = [locationStr2Float(params.trainUsers[params.U_train[nearest_neighbor]]) for nearest_neighbor in nearest_neighbors]
        nnlat = np.median([coor[0] for coor in coordinates])
        nnlon = np.median([coor[1] for coor in coordinates])
        inclass_nearestn_distances.append(distance(lat, lon, nnlat, nnlon))   

        distances.append(distance(lat, lon, medianlat, medianlon))

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    print "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161))
    print
    inclass_acc_at_161 = 100 * len([d for d in inclass_closest_point_distances if d < 161]) / float(len(inclass_closest_point_distances))
    print "In Class closest -- Mean: " + str(int(np.mean(inclass_closest_point_distances))) + " Median: " + str(int(np.median(inclass_closest_point_distances))) + " Acc@161: " + str(int(inclass_acc_at_161))
    print
    nn_acc_at_161 = 100 * len([d for d in inclass_nearestn_distances if d < 161]) / float(len(inclass_nearestn_distances))
    print "In Class Regression -- Mean: " + str(int(np.mean(inclass_nearestn_distances))) + " Median: " + str(int(np.median(inclass_nearestn_distances))) + " Acc@161: " + str(int(nn_acc_at_161))
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

def loss_classification(preds, U_eval):
    
    if len(preds) != len(U_eval): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_eval))
        print "fatal error!"
        sys.exit()
    #check if it is dev or test
    if U_eval[0] in params.U_dev:
        dev = True
    else:
        dev = False
    def closest_neighbour_classes():
        neighbours = defaultdict(dict)
        sorted_neighbours = {}
        for c1 in params.classLatMedian:
            for c2 in params.classLatMedian:
                if c1 == c2:
                    continue
                lat1, lon1 = params.classLatMedian[c1], params.classLonMedian[c1]
                lat2, lon2 = params.classLatMedian[c2], params.classLonMedian[c2]
                dd = distance(lat1, lon1, lat2, lon2)
                neighbours[c1][c2] = dd
        for c, nbrs in neighbours.iteritems():
            sorted_nbrs = sorted(nbrs.items(), key=operator.itemgetter(1))
            sorted_neighbours[c] = sorted_nbrs
        return sorted_neighbours
                
    sorted_neighbours = closest_neighbour_classes()    
    class_clf = train_classification_models()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    inclass_closest_point_distances = []
    global_closest_point_distances = []
    inclass_naivebayes_distances = []
    user_location = {}
    acc = 0.0
    center_of_us = (39.50, -98.35)
    nyc = (40.7127, -74.0059)
    la = (34.0500, -118.2500)
    distances_from_nyc = []
    distances_from_la = []
    distances_from_center = []
    global_closest = False
    for i in range(0, len(preds)):
        user = U_eval[i]
        location = params.userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        user_original_class, minDistance = assignClass(lat, lon)

        prediction = params.categories[preds[i]]
        medianlat = params.classLatMedian[prediction]  
        medianlon = params.classLonMedian[prediction]
        nearby_classes =  [c[0] for c in sorted_neighbours[prediction][0:1]]
        nearby_classes = []
        nearby_classes.append(prediction)
        nearby_classes = set(nearby_classes)
        
        training_points_in_class = {u:locationStr2Float(params.trainUsers[u]) for u, c in params.trainClasses.iteritems() if str(c) in nearby_classes or global_closest}
        u_point_distances = {u:distance(lat, lon, l[0], l[1]) for u, l in training_points_in_class.iteritems()}
        closest_point =  min(u_point_distances, key=u_point_distances.get)
        inclass_closest_distance = u_point_distances[closest_point]
        inclass_closest_point_distances.append(inclass_closest_distance)
        
        #regress within predicted class using knn
        clf, uids, region_median = class_clf[preds[i]]
        if dev:
            X_u = params.X_dev[i]
        else:
            X_u = params.X_test[i]

        region = clf.predict(X_u)[0]
        nblat, nblon = region_median[region]
        inclass_naivebayes_distances.append(distance(lat, lon, nblat, nblon))   

        distances.append(distance(lat, lon, medianlat, medianlon))

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    print "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161))
    print
    inclass_acc_at_161 = 100 * len([d for d in inclass_closest_point_distances if d < 161]) / float(len(inclass_closest_point_distances))
    print "In Class closest -- Mean: " + str(int(np.mean(inclass_closest_point_distances))) + " Median: " + str(int(np.median(inclass_closest_point_distances))) + " Acc@161: " + str(int(inclass_acc_at_161))
    print
    nn_acc_at_161 = 100 * len([d for d in inclass_naivebayes_distances if d < 161]) / float(len(inclass_naivebayes_distances))
    print "In Class classification -- Mean: " + str(int(np.mean(inclass_naivebayes_distances))) + " Median: " + str(int(np.median(inclass_naivebayes_distances))) + " Acc@161: " + str(int(nn_acc_at_161))
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


def loss_latlon(U_eval, predicted_coordinates):
    distances = []
    if U_eval[0] == params.U_dev[0]:
        dev = True
        eval_locations = [locationStr2Float(params.devUsers[params.U_dev[i]]) for i in range(0, len(params.U_dev))]
    elif U_eval[0] == params.U_test[0]:
        dev = False
        eval_locations = [locationStr2Float(params.testUsers[params.U_test[i]]) for i in range(0, len(params.U_test))]
    
    for i in range(0, len(predicted_coordinates)):
        real_lat, real_lon = eval_locations[i]
        pred_lat, pred_lon = predicted_coordinates[i]
        distances.append(distance(real_lat, real_lon, pred_lat, pred_lon))
        

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info( "Mean " + str(int(np.mean(distances))) + ' median ' + str(int(np.median(distances))) + ' acc@161 ' + str(acc_at_161)) 
    
     

  




def feature_extractor(use_mention_dictionary=False, use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=1, max_df=1.0, BuildCostMatrices=False, vectorizer=None, stop_words=None, novectorization=False, vocab=None, save_vectorizer=True, complete_prob=False):
    '''
    read train, dev and test dictionaries and extract textual features using tfidfvectorizer.
    '''
    
    print("%d params.categories" % len(params.categories))
    print()
    # split a training set and a test set
    params.Y_train = np.asarray([params.trainClasses[u] for u in params.U_train])
    params.Y_test = np.asarray([params.testClasses[u] for u in params.U_test])
    params.Y_dev = np.asarray([params.devClasses[u] for u in params.U_dev])


    logging.info("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time.time()

    if vectorizer == None:  
        if use_mention_dictionary:
            print "using @ mention dictionary as vocab..."
            extract_mentions()
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=1, max_df=max_df, ngram_range=(1, 1), vocabulary=params.mentions, stop_words=stop_words, encoding=params.data_encoding)
        else:
            print "mindf: " + str(min_df) + " maxdf: " + str(max_df)
            vectorizer = TfidfVectorizer(token_pattern=r'(?u)[#@]*\b\w\w+\b', use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words=stop_words, vocabulary=vocab, encoding=params.data_encoding)
    print vectorizer
    
    params.X_train = vectorizer.fit_transform([params.trainText[u] for u in params.U_train])
    vectorizer.stop_words_ = None
    if save_vectorizer:
        vectorizer_dump_file = path.join(params.GEOTEXT_HOME, 'vectorizer.pkl')
        logging.info('dumping the vectorizer into ' + vectorizer_dump_file)
        with open(vectorizer_dump_file, 'wb') as outf:
            pickle.dump(vectorizer, outf)
    
    params.feature_names = vectorizer.get_feature_names()
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % params.X_train.shape)
    print()
    
    print("Extracting features from the dev dataset using the same vectorizer")
    t0 = time.time()
    params.X_dev = vectorizer.transform([params.devText[u] for u in params.U_dev])
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % params.X_dev.shape)
    print()

    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time.time()
    params.X_test = vectorizer.transform([params.testText[u] for u in params.U_test])
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % params.X_test.shape)
    print()
    
    if complete_prob:
        logging.info('Building complete probability distribution for target values...')
        Y_train = np.zeros((params.X_train.shape[0], len(params.categories)), dtype='int32')
        Y_dev = np.zeros((params.X_dev.shape[0], len(params.categories)), dtype='int32')
        Y_test = np.zeros((params.X_test.shape[0], len(params.categories)), dtype='int32')
        for i in range(params.Y_dev.shape[0]):
            Y_dev[i, params.Y_dev[i]] = 1
        for i in range(params.Y_test.shape[0]):
            Y_test[i, params.Y_test[i]] = 1
        for i in range(params.Y_train.shape[0]):
            Y_train[i, params.Y_train[i]] = 1
    params.vectorizer = vectorizer 
    return params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names

def get_topk_features(label, topk=50):
    """
    given a label (str) return the top k important features as a list
    """
    topk_feature_indices = np.argsort(params.clf.coef_[label].toarray())[0,-topk:].tolist()[::-1]
    topk_features = [params.vectorizer.features[i] for i in topk_feature_indices]
    topk_weights = [params.clf.coef_[label, i] for i in topk_feature_indices]
    topk_features = [f for f in topk_features if 'user_' not in f]
    return topk_features, topk_weights  
    

    
def classify(granularity=10, DSExpansion=False, DSModification=False, compute_dev=False, report_verbose=False, clf=None, regul=0.00001, partitionMethod='median', penalty=None, fit_intercept=False, save_model=False, reload_model=False):
    model_dump_file = path.join(params.GEOTEXT_HOME, 'model-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE) + '-' + str(regul) + '.pkl')
    top_features_file = path.join(params.GEOTEXT_HOME, 'topfeatures-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE) + '-' + str(regul) + '.txt')
    compute_dev = True

    if clf == None:
        # clf = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l2', dual=False, tol=1e-3)
        # clf = linear_model.LogisticRegression(C=1.0, penalty='l2')
        # alpha = 0.000001
        clf = SGDClassifier(loss='log', alpha=regul, penalty=params.penalty, l1_ratio=0.9, learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=40, fit_intercept=fit_intercept)
        #clf = LogisticRegression(penalty='l2', dual=False, C = 1.0, fit_intercept=fit_intercept)
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
        params.X_train = params.X_train.toarray()
        params.X_test = params.X_test.toarray()
        params.X_dev = params.X_dev.toarray()
    model_reloaded = False
    if reload_model and path.exists(model_dump_file):
        print('loading a trained model from %s' % (model_dump_file))
        with gzip.open(model_dump_file, 'rb') as inf:
            clf = pickle.load(inf)
            model_reloaded = True
        print(clf)
    if not model_reloaded:
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time.time()
        clf.fit(params.X_train, params.Y_train)
        params.clf = clf
        train_time = time.time() - t0
        print("train time: %0.3fs" % train_time)
        if hasattr(clf, 'coef_'):
            zero_count = (clf.coef_ == 0).sum()
            total_param_count = clf.coef_.shape[0] * clf.coef_.shape[1]
            zero_percent = int(100 * float(zero_count) / total_param_count)
            print('%d percent sparse' % (zero_percent))
            if zero_percent > 50:
                print('sparsifying clf.coef_ to free memory')
                clf.sparsify()
        if save_model:
            print('dumpinng the model in %s' % (model_dump_file))
            #joblib.dump(value=clf, filename=model_dump_file, compress=3)
            with open(model_dump_file, 'wb') as outf:
                pickle.dump(clf, outf)
    if hasattr(clf, 'coef_'):
        non_zero_parameters = csr_matrix(clf.coef_).nnz
    else:
        non_zero_parameters = 0

    report_verbose = True
    if compute_dev:
        devPreds = clf.predict(params.X_dev)
        if hasattr(clf, 'predict_proba'):
            devProbs = clf.predict_proba(params.X_dev)
        else:
            devProbs = None

    save_vocabulary = False
    if save_vocabulary:
        rows, cols = np.nonzero(clf.coef_)
        del rows
        vocab = params.feature_names[cols]
        vocab_file = path.join(params.GEOTEXT_HOME, 'vocab.pkl')
        with open(vocab_file, 'wb') as outf:
            pickle.dump(vocab, outf)
    abod = False
    print "predicting test labels"  
    t0 = time.time()
    preds = clf.predict(params.X_test)
    # scores = clf.decision_function(params.X_test)
    if hasattr(clf, 'predict_proba'):
        testProbs = clf.predict_proba(params.X_test)
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
                for i, category in enumerate(params.categories):
                    top10 = np.argsort(clf.coef_[i])[-50:]
                    # print("%s: %s" % (category, " ".join(params.feature_names[top10])))
                    outf.write(category + ": " + " ".join(params.feature_names[top10]) + '\n')


    
    print "test results"
    meanTest, medianTest, acc_at_161_test = loss(preds, params.U_test)
    meanDev = -1
    medianDev = -1
    if compute_dev:
        print "development results"
        meanDev, medianDev, acc_at_161_dev = loss(devPreds, params.U_dev)

    #print 'test'
    #loss_regression(preds, params.U_test)
    #print 'test lr'
    #loss_classification(preds, params.U_test)
    #print 'test gmm'
    #loss_clustering(preds, params.U_test)
    #print 'dev'
    #loss_regression(devPreds, U_eval=params.U_dev)
    #print 'dev lr'
    #loss_classification(devPreds, U_eval=params.U_dev)
    #print 'dev gmm'
    #loss_clustering(devPreds, U_eval=params.U_dev)
    dump_preds = False
    if dump_preds:
        result_dump_file = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE) + '-lr.pkl')
        print "dumping preds (preds, devPreds, params.U_test, params.U_dev, testProbs, devProbs) in " + result_dump_file
        with open(result_dump_file, 'wb') as outf:
            pickle.dump((preds, devPreds, params.U_test, params.U_dev, testProbs, devProbs), outf)
    # evaluate(preds,params.U_test, params.categories, None)
    # abod(probs, preds, params.U_test)
    return preds, probs, params.U_test, meanTest, medianTest, acc_at_161_test, meanDev, medianDev, acc_at_161_dev, non_zero_parameters
   
# classify()









def initialize(granularity, write=False, readText=True, reload_init=False, regression=False):    

    reload_file = path.join(params.GEOTEXT_HOME + '/init_' + params.DATASETS[params.DATASET_NUMBER - 1] + '_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + '.pkl')

    logging.info('reading (user_info.) train, dev and test file and building params.trainUsers, params.devUsers and params.testUsers with their locations')
    if params.TEXT_ONLY:
        logging.info('mentions are excluded from features.')
    users(params.trainfile, 'train', write, readText=readText)
    users(params.devfile, 'dev', write, readText=readText)
    users(params.testfile, 'test', write, readText=readText)
    logging.info("the number of train" + " users is " + str(len(params.trainUsers)))
    logging.info("the number of test" + " users is " + str(len(params.testUsers)))
    logging.info("the number of dev" + " users is " + str(len(params.devUsers)))
    

    logging.info('discretization method: ' + params.partitionMethod)
    if not regression:
        if params.partitionMethod == 'median':
            discretize_kdtree(granularity, params.partitionMethod)
        elif params.partitionMethod == 'kmeans':
            descritize_users(n_clusters=params.num_class) 
        write_init_info = False
        if write_init_info:
            logging.info('writing init info in %s' % (reload_file))
            with open(reload_file, 'wb') as outf:
                pickle.dump((params.classLatMedian, params.classLonMedian), outf)
    else:
        logging.info('Not discretising locations as regression option is on!')
    logging.info("initialization finished")


def factorize(X, transformees, factorizer=None):
    if factorizer is None:
        factorizer = DictionaryLearning(n_components=100, alpha=1, max_iter=100, tol=1e-8, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=30, code_init=None, dict_init=None, verbose=True, split_sign=None, random_state=None)
    else:
        print factorizer
    # dic_learner = DictionaryLearning(params.n_components=100, alpha=1, max_iter=100, tol=1e-8, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=30, code_init=None, dict_init=None, verbose=True, split_sign=None, random_state=None)
    # dic_learner = MiniBatchDictionaryLearning(params.n_components=20, alpha=1, n_iter=100, fit_algorithm='lars', n_jobs=30, batch_size=1000, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=True, split_sign=True, random_state=None)
    # dic_learner = PCA(params.n_components=500)
    
    
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

def asclassification(granularity,  use_mention_dictionary=False, use_sparse_code=False, penalty=None, fit_intercept=False, norm=None, binary=False, sublinear=False, factorizer=None, read_vocab=False, use_idf=True):

    if read_vocab:
        vocab_file = path.join(params.GEOTEXT_HOME, 'vocab.pkl')
        with open(vocab_file, 'rb') as inf:
            vocab = pickle.load(inf)
            vocab = list(set(vocab))
    else:
        vocab = None
    stops = 'english'
    # partitionLocView(granularity=granularity, params.partitionMethod=params.partitionMethod)
    params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names = feature_extractor(norm=norm, use_mention_dictionary=use_mention_dictionary, min_df=10, max_df=0.2, stop_words=stops, binary=binary, sublinear_tf=sublinear, vocab=vocab, use_idf=use_idf, save_vectorizer=False)    
    if use_sparse_code:
        print("using a matrix factorization technique to learn a better representation of users...")
        sparse_coded_dump = path.join(params.GEOTEXT_HOME, 'sparse_coded.pkl')
        if os.path.exists(sparse_coded_dump) and params.DATASET_NUMBER != 1:
            with open(sparse_coded_dump, 'rb') as inf:
                params.X_train, params.X_dev, params.X_test = pickle.load(inf)
        else:
            params.X_train, params.X_dev, params.X_test = factorize(params.X_train, transformees=[params.X_train, params.X_dev, params.X_test], factorizer=factorizer)
            save_sparse = False
            if save_sparse:
                with open(sparse_coded_dump, 'wb') as inf:
                    pickle.dump((params.X_train, params.X_dev, params.X_test), inf)
    
    best_dev_acc = -1
    best_dev_mean = -1
    best_dev_median = -1
    best_regul = -1
    best_test_preds = None
    regul_acc = {}
    regul_nonzero = {}
    cv_tune = False
    if cv_tune:
        from sklearn.grid_search import GridSearchCV
        clf = SGDClassifier(loss='log', penalty=penalty, l1_ratio=0.9, learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=5, fit_intercept=fit_intercept)
        tuned_parameters = {'alpha': [1e-6, 5e-5, 1e-5]}
        clf.fit(params.X_train, params.Y_train)
        clf = GridSearchCV(clf, tuned_parameters, cv=5, scoring='log_loss', n_jobs=10)
        best_test_preds = clf.predict(params.X_test)
        best_dev_preds = clf.predict(params.X_dev)
        print 'test results'
        t_mean, t_median, t_acc = loss(best_test_preds, params.U_test)
        print 'dev results'
        t_mean, t_median, t_acc = loss(best_dev_preds, params.U_dev)
        
    else:
        
        if params.DATASET_NUMBER == 1:
            reguls_coefs = [5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 4e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-2, 1e-2]
        elif params.DATASET_NUMBER == 2:
            reguls_coefs = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
            reguls_coefs = [2 ** a for a in xrange(-27, -3, 2)]
            if params.BUCKET_SIZE <= 1024:
                reguls_coefs = [2 ** -21]
            else:
                reguls_coefs = [2 ** -19]
        elif params.DATASET_NUMBER == 3:
            reguls_coefs = [9e-8, 1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7]
        elif params.DATASET_NUMBER == 4:
            reguls_coefs = [1e-7, 2e-7, 4e-7, 6e-7, 8e-7, 1e-6, 2e-6, 4e-6]
        if penalty == 'none':
            reguls_coefs = [1e-200]
        #for regul in reguls_coefs:
        for regul in [params.reguls[params.DATASET_NUMBER - 1]]:
            preds, probs, params.U_test, meanTest, medianTest, acc_at_161_test, meanDev, medianDev, acc_at_161_dev, non_zero_parameters = classify(granularity=granularity, regul=regul, penalty=penalty, fit_intercept=fit_intercept, reload_model=False, save_model=False)
            regul_acc[regul] = acc_at_161_dev
            regul_nonzero[regul] = non_zero_parameters
            if acc_at_161_dev > best_dev_acc:
                best_dev_acc = acc_at_161_dev
                best_regul = regul
                best_test_preds = preds
                best_dev_mean = meanDev
                best_dev_median = medianDev
                
        print('The best regul_coef is %e %f' % (best_regul, best_dev_acc))
    t_mean, t_median, t_acc = loss(best_test_preds, params.U_test)
    
    return t_mean, t_median, t_acc, best_dev_mean, best_dev_median, best_dev_acc
    # return preds, probs, params.U_test, meanTest, medianTest, acc_at_161_test, meanDev, medianDev, acc_at_161_dev


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
    # if it is there load it and return
    mention_file_address = path.join(params.GEOTEXT_HOME, 'params.mentions.pkl')
    if addDev:
        mention_file_address = mention_file_address + '.dev'
    RELOAD_mentions = False
    if RELOAD_mentions:
        if os.path.exists(mention_file_address):
            print "reading params.mentions from pickle"
            with open(mention_file_address, 'rb') as inf:
                params.mentions = pickle.load(inf)
                return
    text = ''
    # for user in params.trainUsers:
    #    text += userText[user].lower()
    if addTest:
        text = ' '.join(params.trainText.values() + params.testText.values())
    if addDev:
        text = ' '.join(params.trainText.values() + params.devText.values())
    if not addTest and not addDev:
        text = ' '.join(params.trainText.values())
    # text = text.lower()
    '''
    if params.data_encoding in ['utf-8']:
        text = strip_accents_unicode(text)
    elif params.data_encoding in ['latin', 'latin1']:
        text = strip_accents_ascii(text)
    '''
    token_pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern = re.compile(token_pattern)
    print "finding params.mentions"
    mentionsList = [word.lower() for word in token_pattern.findall(text)]
    print "building the counter"
    mentionsDic = Counter(mentionsList)
    print "frequency thresholding"
    if k > 0:
        params.mentions = [word for word in mentionsDic if mentionsDic[word] > k]
    else:
        params.mentions = mentionsDic.keys()

def prepare_adsorption_data_collapsed(DEVELOPMENT=False, text_prior=params.prior, CELEBRITY_THRESHOLD=100000, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=False):
    MULTI_LABEL = False

    dongle_nodes = None
    dongle_preds = None
    dongle_probs = None
    params.U_train = [u for u in sorted(params.trainUsers)]
    params.U_test = [u for u in sorted(params.testUsers)]
    params.U_dev = [u for u in sorted(params.devUsers)]
    text_str = ''
    if text_prior != 'none':
        text_str = '.' + text_prior
    weighted_str = ''
    if DIRECT_GRAPH_WEIGHTED:
        weighted_str = '.weighted'
    
    celebrityStr = str(CELEBRITY_THRESHOLD)

    
    # split a training set and a test set
    params.Y_train = np.asarray([params.trainClasses[u] for u in params.U_train])
    # params.Y_test = np.asarray([params.testClasses[u] for u in params.U_test])
    # params.Y_dev = np.asarray([params.devClasses[u] for u in params.U_dev])

    
    if DEVELOPMENT:
        devStr = '.dev'
    else:
        devStr = ''

    
    if text_prior != 'none':
        logging.info("tex params.prior is " + text_prior)
        # read users and predictions
        result_dump_file = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE) + '-' + params.textmodel + '.pkl')
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
              

    id_user_file = path.join(params.GEOTEXT_HOME, 'id_user_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing id_user in " + id_user_file)
    with codecs.open(id_user_file, 'w', 'ascii') as outf:
        for i in range(0, len(params.U_train)):
            outf.write(str(i) + '\t' + params.U_train[i] + '\n')
        if DEVELOPMENT:
            for i in range(0, len(params.U_dev)):
                outf.write(str(i + len(params.U_train)) + '\t' + params.U_dev[i] + '\n')
        else:
            for i in range(0, len(params.U_test)):
                outf.write(str(i + len(params.U_train)) + '\t' + params.U_test[i] + '\n')

    seed_file = path.join(params.GEOTEXT_HOME, 'seeds_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing seeds in " + seed_file)
    with codecs.open(seed_file, 'w', 'ascii') as outf:
        for i in range(0, len(params.U_train)):
            outf.write(str(i) + '\t' + str(params.Y_train[i]) + '\t' + '1.0' + '\n')

        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                # w = np.max(dongle_probs[i])
                w = 1
                outf.write(str(i + len(params.U_train)) + '.T' + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'direct':
            w = np.max(dongle_probs[i])
            outf.write(str(i + len(params.U_train)) + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'backoff':
            pass 

    doubles = 0
    double_nodes = []
    trainIdx = range(len(params.U_train))
    trainUsersLowerDic = dict(zip(params.U_train, trainIdx))
    if DEVELOPMENT:
        devStr = '.dev'
        for i in range(0, len(params.U_dev)):
            u = params.U_dev[i]
            if u in trainUsersLowerDic:
                params.U_dev[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_unknown = params.U_dev
        u_text_unknown = params.devText
    else:
        for i in range(0, len(params.U_test)):
            u = params.U_test[i]
            if u in trainUsersLowerDic:
                params.U_test[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_text_unknown = params.testText
        u_unknown = params.U_test
    if text_prior != 'none':
        assert(len(u_unknown) == len(dongle_nodes)), 'the number of text/dev users is not equal to the number of text predictions.'
    U_all = params.U_train + u_unknown 

    logging.info("The number of test users found in train users is " + str(doubles))
    logging.info("Double users: " + str(double_nodes))
    vocab_cnt = len(U_all)
    idx = range(vocab_cnt)
    node_id = dict(zip(U_all, idx))
    # data and indices of a coo matrix to be populated
    coordinates = Counter()
    
    # for node, id in node_id.iteritems():
    #    node_lower_id[node.lower()] = id
    assert (len(node_id) == len(params.U_train) + len(u_unknown)), 'number of unique users is not eq params.U_train + params.U_test'
    logging.info("the number of nodes is " + str(vocab_cnt))
    logging.info("Adding the direct relationships...")
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    mention_users = defaultdict(Counter)
    directly_connected_users = set()
    l = len(params.trainText)
    tenpercent = l / 10
    i = 1
    for user, text in params.trainText.iteritems():
        user_id = node_id[user]
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        params.mentions = [u.lower() for u in token_pattern1.findall(text)] 
        mentionDic = Counter(params.mentions)
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
        params.mentions = [u.lower() for u in token_pattern1.findall(text)]
        mentionDic = Counter(params.mentions)
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
    input_graph_file = path.join(params.GEOTEXT_HOME, 'input_graph_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + '_' + celebrityStr + devStr + text_str + weighted_str)
    logging.info("writing the input_graph in " + input_graph_file + ' edges weighted (not binary): ' + str(DIRECT_GRAPH_WEIGHTED))
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
                params.mention_graph.add_edge(xindx, yindx, attr_dict={'w':w})
        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                confidence = np.max(dongle_probs[i])
                outf.write(str(i + len(params.U_train)) + '.T' + '\t' + str(i + len(params.U_train)) + '\t' + str(confidence) + '\n')
        elif text_prior == 'direct':
            pass
        elif text_prior == 'backoff':
            pass
    
    u1s = [a for a, b in coordinates]
    u2s = [b for a, b in coordinates]
    us = set(u1s + u2s)
    logging.info("the number of disconnected users is " + str(len(U_all) - len(us)))
    logging.info("the number of disconnected test users is " + str(len(u_unknown) - len([a for a in us if a >= len(params.U_train)])))
    output_file = path.join(params.GEOTEXT_HOME, 'label_prop_output_' + params.DATASETS[params.DATASET_NUMBER - 1] + '_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + '_' + str(params.celeb_threshold) + devStr + text_str + weighted_str)
    logging.info("output file: " + output_file)
    # logging.info(str(disconnected_us))

def prepare_adsorption_data_collapsed_networkx(DEVELOPMENT=False, text_prior='none', CELEBRITY_THRESHOLD=100000, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=True, postfix='.nx'):

    MULTI_LABEL = False

    dongle_nodes = None
    dongle_preds = None
    dongle_probs = None
    params.U_train = [u for u in sorted(params.trainUsers)]
    params.U_test = [u for u in sorted(params.testUsers)]
    params.U_dev = [u for u in sorted(params.devUsers)]
    text_str = ''
    if text_prior != 'none':
        text_str = '.' + text_prior
    weighted_str = ''
    if DIRECT_GRAPH_WEIGHTED:
        weighted_str = '.weighted'
    
    celebrityStr = str(CELEBRITY_THRESHOLD)
    params.mention_graph = nx.Graph()

    params.Y_train = np.asarray([params.trainClasses[u] for u in params.U_train])

    
    if DEVELOPMENT:
        devStr = '.dev'
    else:
        devStr = ''



    
    if text_prior != 'none':
        logging.info("tex params.prior is " + text_prior)
        # read users and predictions
        result_dump_file = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE)  + '-' + params.textmodel + '.pkl')
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
              
    id_user_file = path.join(params.GEOTEXT_HOME, 'id_user_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing id_user in " + id_user_file)
    with codecs.open(id_user_file, 'w', 'ascii') as outf:
        for i in range(0, len(params.U_train)):
            outf.write(str(i) + '\t' + params.U_train[i] + '\n')
        if DEVELOPMENT:
            for i in range(0, len(params.U_dev)):
                outf.write(str(i + len(params.U_train)) + '\t' + params.U_dev[i] + '\n')
        else:
            for i in range(0, len(params.U_test)):
                outf.write(str(i + len(params.U_train)) + '\t' + params.U_test[i] + '\n')

    seed_file = path.join(params.GEOTEXT_HOME, 'seeds_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + devStr + text_str + weighted_str)
    logging.info("writing seeds in " + seed_file)
    with codecs.open(seed_file, 'w', 'ascii') as outf:
        for i in range(0, len(params.U_train)):
            outf.write(str(i) + '\t' + str(params.Y_train[i]) + '\t' + '1.0' + '\n')

        if text_prior == 'dongle':
            for i in range(0, len(dongle_nodes)):
                # w = np.max(dongle_probs[i])
                w = 1
                outf.write(str(i + len(params.U_train)) + '.T' + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'direct':
            w = np.max(dongle_probs[i])
            outf.write(str(i + len(params.U_train)) + '\t' + str(dongle_preds[i]) + '\t' + str(w) + '\n')
        elif text_prior == 'backoff':
            pass 

    doubles = 0
    double_nodes = []
    trainIdx = range(len(params.U_train))
    trainUsersLowerDic = dict(zip(params.U_train, trainIdx))
    if DEVELOPMENT:
        devStr = '.dev'
        for i in range(0, len(params.U_dev)):
            u = params.U_dev[i]
            if u in trainUsersLowerDic:
                params.U_dev[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_unknown = params.U_dev
        u_text_unknown = params.devText
    else:
        for i in range(0, len(params.U_test)):
            u = params.U_test[i]
            if u in trainUsersLowerDic:
                params.U_test[i] = u + '_double00'
                double_nodes.append(u)
                doubles += 1
        u_text_unknown = params.testText
        u_unknown = params.U_test
    if text_prior != 'none':
        assert(len(u_unknown) == len(dongle_nodes)), 'the number of text/dev users is not equal to the number of text predictions.'
    
    
    
    U_all = params.U_train + u_unknown 
    logging.info("The number of test users found in train users is " + str(doubles))
    logging.info("Double users: " + str(double_nodes))
    vocab_cnt = len(U_all)
    idx = range(vocab_cnt)
    node_id = dict(zip(U_all, idx))
    idx = list(idx)
    params.mention_graph.add_nodes_from(idx)
    # for node, id in node_id.iteritems():
    #    node_lower_id[node.lower()] = id
    assert (len(node_id) == len(params.U_train) + len(u_unknown)), 'number of unique users is not eq params.U_train + params.U_test'
    logging.info("the number of nodes is " + str(vocab_cnt))
    logging.info("Adding the direct relationships...")
    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    l = len(params.trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    params.mention_graph.add_nodes_from(node_id.values())
    for user, text in params.trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id] 
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not DIRECT_GRAPH_WEIGHTED:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in u_text_unknown.iteritems():
        user_id = node_id[user]
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id]
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not DIRECT_GRAPH_WEIGHTED:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)  
    
    
    celebrities = []
    remove_celebrities = True
    if remove_celebrities:
        nodes = params.mention_graph.nodes_iter()
        for node in nodes:
            nbrs = params.mention_graph.neighbors(node)
            if len(nbrs) > CELEBRITY_THRESHOLD:
                celebrities.append(node)
        celebrities = [c for c in celebrities if c not in idx]
        logging.info("found %d celebrities with celebrity threshold %d" % (len(celebrities), CELEBRITY_THRESHOLD))
        for celebrity in celebrities:
                params.mention_graph.remove_node(celebrity)
    
    project_to_main_users = True
    if project_to_main_users:
        # params.mention_graph = bipartite.overlap_weighted_projected_graph(params.mention_graph, main_users, jaccard=False)
        params.mention_graph = efficient_collaboration_weighted_projected_graph(params.mention_graph, idx, weight_str=None, degree_power=1, caller='mad')
        # params.mention_graph = collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str='weight')
        # params.mention_graph = bipartite.projected_graph(params.mention_graph, main_users)
    connected_nodes = set()
    l = params.mention_graph.number_of_edges()
    logging.info("The number of edges is " + str(l))
    tenpercent = l / 10
    input_graph_file = path.join(params.GEOTEXT_HOME, 'input_graph_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + '_' + celebrityStr + devStr + text_str + weighted_str + postfix)
    logging.info("writing the input_graph in " + input_graph_file)
    with codecs.open(input_graph_file, 'w', 'ascii', buffering=pow(2, 6) * pow(2, 20)) as outf:
        i = 1
        for edge in params.mention_graph.edges_iter(nbunch=None, data=True):
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
                outf.write(str(i + len(params.U_train)) + '.T' + '\t' + str(i + len(params.U_train)) + '\t' + str(confidence) + '\n')
        elif text_prior == 'direct':
            pass
        elif text_prior == 'backoff':
            pass
    
    
    us = [a for a in idx if a in connected_nodes]
    logging.info("the number of disconnected users is " + str(len(U_all) - len(us)))
    logging.info("the number of disconnected test users is " + str(len(u_unknown) - len([a for a in us if a >= len(params.U_train)])))
    output_file = path.join(params.GEOTEXT_HOME, 'label_prop_output_' + params.DATASETS[params.DATASET_NUMBER - 1] + '_' + params.partitionMethod + '_' + str(params.BUCKET_SIZE) + '_' + str(params.celeb_threshold) + devStr + text_str + weighted_str)
    logging.info("output file: " + output_file)
    # logging.info(str(disconnected_us))
    


def ideal_network_errors():
    graph_file_address = path.join(params.GEOTEXT_HOME, 'direct_graph')
    if os.path.exists(graph_file_address):
        print "reading netgraph from pickle"
        with open(graph_file_address, 'rb') as inf:
            netgraph, params.trainUsers, params.testUsers = pickle.load(inf)
    ideal_distances = []
    acc161 = 0
    tenpercent = len(params.testUsers) / 10
    i = 0
    for utest, uloc in params.testUsers.iteritems():
        i += 1
        if i % tenpercent == 0:
            print str(100 * i / len(params.testUsers))
        lat1, lon1 = locationStr2Float(uloc)
        dists = []
        for utrain, utrainloc in params.trainUsers.iteritems():
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
    


                
def LP(weighted=True, normalize_edge=False, remove_celebrities=False, dev=False, node_order='l2h', remove_mentions_with_degree_one=False):
    '''
    Run label propagation over real-valued coordinates of the @-mention graph.
    The coordinates of the training users are kept unchanged and the coordinates
    of other users are updated to the median of their neighbours.
    '''
    if dev:
        evalText = params.devText
        evalUsers = params.devUsers
    else:
        evalText = params.testText
        evalUsers = params.testUsers

    params.mention_graph = nx.Graph()
    U_all = params.trainUsers.keys() + evalUsers.keys()
    assert len(U_all) == len(params.trainUsers) + len(evalUsers), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    
    print('weighted=%s and params.prior=%s' % (weighted, params.prior))
    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    netgraph = {}
    l = len(params.trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    params.mention_graph.add_nodes_from([(u, {'train':True, 'loc':l}) for u, l in params.trainUsers.iteritems()])
    params.mention_graph.add_nodes_from([(u, {'test':True, 'loc':l}) for u, l in evalUsers.iteritems()])
    for user, text in params.trainText.iteritems():    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        params.mentions = [u.lower() for u in token_pattern1.findall(text)] 
        params.mentions = [m for m in params.mentions if m != user]
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user, mention):
                params.mention_graph[user][mention]['weight'] += freq
            else:
                params.mention_graph.add_edge(user, mention, weight=freq)   
        
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        params.mentions = [u.lower() for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user]
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user, mention):
                params.mention_graph[user][mention]['weight'] += freq
            else:
                params.mention_graph.add_edge(user, mention, weight=freq)  
    

    lp_graph = params.mention_graph          
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
    if params.prior != 'none':
        if params.prior == 'backoff':
            backoff = True
        elif params.prior == 'prior':
            text_direct = True
        elif params.prior == 'dongle':
            dongle = True
        prior_file_path = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE)  + '-' + params.textmodel + '.pkl')
        print "reading params.prior text-based locations from " + prior_file_path
        if os.path.exists(prior_file_path):
            with open(prior_file_path, 'rb') as inf:
                preds, devPreds, params.U_test, params.U_dev, testProbs, devProbs = pickle.load(inf)
                if dev:
                    preds = devPreds
                    params.U_test = params.U_dev
                    testProbs = devProbs
                test_confidences = testProbs[np.arange(0, preds.shape[0]), preds]
                loss(preds=preds, U_test=params.U_test)
                if dongle:
                    params.mention_graph.add_nodes_from([u + '.dongle' for u in params.U_test])
                user_index = 0   
                for user, pred in zip(params.U_test, preds):
                    lat = params.classLatMedian[str(pred)]
                    lon = params.classLonMedian[str(pred)]
                    if backoff:
                        text_preds[user] = (lat, lon)
                    if dongle:
                        dongle_node = user + '.dongle'
                        w = test_confidences[user_index]
                        params.mention_graph.add_edge(dongle_node, user, weight=w)
                        node_location[dongle_node] = (lat, lon)
                        dongle_nodes.append(dongle_node)
                    elif text_direct:
                        node_location[user] = (lat, lon)
                    user_index += 1
        else:
            print "prior file not found."
    for user, loc in params.trainUsers.iteritems():
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)
        trainUsersLower[user] = (lat, lon)
        node_location[user] = (lat, lon)
        
    for user, loc in evalUsers.iteritems():
        lat, lon = locationStr2Float(loc)
        evalUsersLower[user] = (lat, lon)
    
    lp_graph = evalUsersLower
    print "the number of train nodes is " + str(len(params.trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    medianLat = np.median(trainLats)
    medianLon = np.median(trainLons)

    # remove celebrities from the graph
    
    remove_betweeners = False
    if remove_betweeners:
        print("computing betweenness centrality of all nodes, takes a long time, sorry!")
        scores = nx.betweenness_centrality(params.mention_graph, weight='weight')
        i = 0
        percent_5 = len(scores) / 20 
        for w in sorted(scores, key=scores.get, reverse=True):
            i += 1
            if i < percent_5:
                params.mention_graph.remove_node(w)
    
    
    celebrity_threshold = params.celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = params.mention_graph.nodes_iter()
        for node in nodes:
            nbrs = params.mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in evalUsersLower and node not in trainUsersLower:
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
            params.mention_graph.remove_node(celebrity)
 
    if remove_mentions_with_degree_one:
        mention_nodes = set(params.mention_graph.nodes()) - set(U_all) - set(dongle_nodes)
        mention_degree = params.mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = [node for node, degree in mention_degree.iteritems() if degree < 2]
        logging.info('found ' + str(len(one_degree_non_target)) + ' params.mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            params.mention_graph.remove_node(node)
    print "finding unlocated nodes"
    if node_order:
        node_degree = params.mention_graph.degree()
        # sort node_degree by value
        if node_order == 'h2l':
            reverse_order = True
        else:
            reverse_order = False
        nodes = sorted(node_degree, key=node_degree.get, reverse=reverse_order)
        if node_order == 'random':
            random.shuffle(nodes)
        # nodes_unknown = [node for node in params.mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]
        nodes_unknown = [node for node in nodes if node not in trainUsersLower and node not in dongle_nodes]
    else:
        nodes_unknown = [node for node in params.mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]

    # find the cycles with 3 nodes and increase their edge weight
    increase_cyclic_edge_weights = False
    if increase_cyclic_edge_weights:
        increase_coefficient = 2
        cycls_3 = [c for c in list(nx.find_cliques(params.mention_graph)) if len(c) > 2]
        print(str(len(cycls_3)) + ' triangles in the graph.')
        # cycls_3 = [c for c in nx.cycle_basis(params.mention_graph) if len(c)==3]
        for c_3 in cycls_3:
            params.mention_graph[c_3[0]][c_3[1]]['weight'] *= increase_coefficient
            params.mention_graph[c_3[0]][c_3[2]]['weight'] *= increase_coefficient
            params.mention_graph[c_3[1]][c_3[2]]['weight'] *= increase_coefficient
        del cycls_3
    
    # remove (or decrease the weights of) the edges between training nodes which are very far from each other
    remove_inconsistent_edges = False
    if remove_inconsistent_edges:
        max_acceptable_distance = 161
        num_nodes_removed = 0
        edges = params.mention_graph.edges()
        edges = [(a, b) for a, b in edges if a in node_location and b in node_location]
        for node1, latlon1 in node_location.iteritems():
            for node2, latlon2 in node_location.iteritems():
                lat1, lon1 = latlon1
                lat2, lon2 = latlon2
                dd = distance(lat1, lon1, lat2, lon2)
                if dd > max_acceptable_distance:
                    if ((node1, node2) in edges or (node2, node1) in edges):
                        try:
                            params.mention_graph.remove_edge(node1, node2)
                            num_nodes_removed += 1
                        except:
                            pass
        print(str(num_nodes_removed) + ' edges removed from the graph') 
                
    use_shortest_paths = False
    shortest_paths = {}
    if use_shortest_paths:
        shortest_paths = nx.all_pairs_shortest_path_length(params.mention_graph, cutoff=3)
    

    
    logging.info("Edge number: " + str(params.mention_graph.number_of_edges()))
    logging.info("Node number: " + str(params.mention_graph.number_of_nodes()))
    
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
            nbrs = params.mention_graph[node]
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
                    edge_weight = params.mention_graph[node][nbr]['weight']
                    nbrlats.append(lat)
                    nbrlons.append(lon)
                    if normalize_edge and weighted:
                        edge_weight_normalized = float(edge_weight * edge_weight) / (params.mention_graph.degree(nbr) * params.mention_graph.degree(node))
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
        if iter_num > 3:
            with open('node_location_withdeg1_' + str(iter_num) + '.pkl', 'wb') as outf:
                pickle.dump(node_location, outf)
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
    

def LP_collapsed(weighted=True, normalize_edge=False, remove_celebrities=False, dev=False, project_to_main_users=False, node_order='l2h', remove_mentions_with_degree_one=True):
    '''
    runs label propagation over the collapsed @-mention graph.
    In a collapsed @-mention graph, @-params.mentions which are not a member of target nodes (training,dev or test users) are
    removed from the graph. The target nodes which were previously connected through the removed @-mention nodes
    will be connected with an edge.
    If project_to_main_users is False the graph won't be collapsed and the complete @-mention graph will be used.
    '''
    params.U_train = [u for u in sorted(params.trainUsers)]
    params.U_test = [u for u in sorted(params.testUsers)]
    params.U_dev = [u for u in sorted(params.devUsers)]
    U_eval = []
    if dev:
        evalText = params.devText
        evalUsers = params.devUsers
        U_eval = params.U_dev
    else:
        evalText = params.testText
        evalUsers = params.testUsers
        U_eval = params.U_test

    U_all = params.U_train + U_eval
    assert len(U_all) == len(params.U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    
    params.mention_graph = nx.Graph()
    
    print('weighted=%s and params.prior=%s' % (weighted, params.prior))

    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    l = len(params.trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    params.mention_graph.add_nodes_from(node_id.values())
    for user, text in params.trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id] 
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        user_id = node_id[user]
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id]
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)  
        
    
    trainuserid_location = {}
    evaluserid_location = {}
    trainLats = []
    trainLons = []
    node_location = {}
    dongle_nodes = []
    text_preds = {}
    for user, loc in params.trainUsers.iteritems():
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
    
    print "the number of train nodes is " + str(len(params.trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    medianLat = np.median(trainLats)
    medianLon = np.median(trainLons)

    celebrity_threshold = params.celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = params.mention_graph.nodes_iter()
        for node in nodes:
            nbrs = params.mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in evaluserid_location and node not in trainuserid_location:
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
                params.mention_graph.remove_node(celebrity)
    
    dongle = False
    backoff = False
    text_direct = False
    if params.prior != 'none':
        if params.prior == 'backoff':
            backoff = True
        elif params.prior == 'prior':
            text_direct = True
        elif params.prior == 'dongle':
            dongle = True
        prior_file_path = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + params.partitionMethod + '-' + str(params.BUCKET_SIZE)  + '-' + params.textmodel + '.pkl')
        print "reading params.prior text-based locations from " + prior_file_path
        if os.path.exists(prior_file_path):
            with open(prior_file_path, 'rb') as inf:
                preds, devPreds, U_test, U_dev, testProbs, devProbs = pickle.load(inf)
                logging.info('text-based test performance:')
                loss(preds, U_test)
                logging.info('text-based dev performance:')
                loss(devPreds, U_dev)

                if dev:
                    preds = devPreds
                    U_test = U_dev
                    testProbs = devProbs
                test_confidences = testProbs[np.arange(0, preds.shape[0]), preds]
                if dongle:
                    params.mention_graph.add_nodes_from([str(node_id[u]) + '.dongle' for u in U_test])
                user_index = 0   
                for user, pred in zip(params.U_test, preds):
                    user_id = node_id[user]
                    lat = params.classLatMedian[str(pred)]
                    lon = params.classLonMedian[str(pred)]
                    if backoff:
                        text_preds[user_id] = (lat, lon)
                    if dongle:
                        dongle_node = str(user_id) + '.dongle'
                        w = test_confidences[user_index]
                        params.mention_graph.add_edge(dongle_node, user_id, weight=w)
                        node_location[dongle_node] = (lat, lon)
                        dongle_nodes.append(dongle_node)
                    elif text_direct:
                        node_location[user_id] = (lat, lon)
                    user_index += 1
        else:
            print "params.prior file not found."




    if remove_mentions_with_degree_one:
        mention_nodes = set(params.mention_graph.nodes()) - set(node_id.values())
        mention_degree = params.mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = {node for node, degree in mention_degree.iteritems() if degree < 2}
        logging.info('found ' + str(len(one_degree_non_target)) + ' params.mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            params.mention_graph.remove_node(node)
            
    if project_to_main_users:
        logging.info('projecting the graph into the target user.')
        main_users = node_id.values()
        if params.prior == 'dongle':
            main_users += [str(user_id) + '.dongle' for user_id in U_test]
        # params.mention_graph = bipartite.overlap_weighted_projected_graph(params.mention_graph, main_users, jaccard=False)
        #params.mention_graph = collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str=None, degree_power=1, caller='lp')
        params.mention_graph = efficient_collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str=None, degree_power=1, caller='lp')
        # params.mention_graph = collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str='weight')
        # params.mention_graph = bipartite.projected_graph(params.mention_graph, main_users)
    logging.info("Edge number: " + str(params.mention_graph.number_of_edges()))
    logging.info("Node number: " + str(params.mention_graph.number_of_nodes()))
    # results[str(project_to_main_users)] = params.mention_graph.degree().values()
    return
    remove_betweeners = False
    if remove_betweeners:
        print("computing betweenness centrality of all nodes, takes a long time, sorry!")
        scores = nx.betweenness_centrality(params.mention_graph, weight='weight')
        i = 0
        percent_5 = len(scores) / 20 
        for w in sorted(scores, key=scores.get, reverse=True):
            i += 1
            if i < percent_5:
                params.mention_graph.remove_node(w)
    
    
        
    
    # check the effect of geolocation order on performance. The intuition is that
    # if the high confident nodes are geolocated first it may be better because noisy
    # predictions won't propagate from the first iterations.
    if node_order:
        node_degree = params.mention_graph.degree()
        if node_order == 'h2l':
            reverse_order = True
        else:
            reverse_order = False
        nodes = sorted(node_degree, key=node_degree.get, reverse=reverse_order)
        if node_order == 'random':
            random.shuffle(nodes)
        # nodes_unknown = [node for node in params.mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]
        nodes_unknown = [node for node in nodes if node not in trainuserid_location and node not in dongle_nodes]
    else:
        nodes_unknown = [node for node in params.mention_graph.nodes() if node not in trainuserid_location and node not in dongle_nodes]
    # find the cycles with 3 nodes and increase their edge weight
    increase_cyclic_edge_weights = False
    if increase_cyclic_edge_weights:
        increase_coefficient = 2
        cycls_3 = [c for c in list(nx.find_cliques(params.mention_graph)) if len(c) > 2]
        print(str(len(cycls_3)) + ' triangles in the graph.')
        # cycls_3 = [c for c in nx.cycle_basis(params.mention_graph) if len(c)==3]
        for c_3 in cycls_3:
            params.mention_graph[c_3[0]][c_3[1]]['weight'] *= increase_coefficient
            params.mention_graph[c_3[0]][c_3[2]]['weight'] *= increase_coefficient
            params.mention_graph[c_3[1]][c_3[2]]['weight'] *= increase_coefficient
        del cycls_3
    
    # remove (or decrease the weights of) the edges between training nodes which are very far from each other
    remove_inconsistent_edges = False
    if remove_inconsistent_edges:
        max_acceptable_distance = 161
        num_nodes_removed = 0
        edges = params.mention_graph.edges()
        edges = [(a, b) for a, b in edges if a in node_location and b in node_location]
        for node1, latlon1 in node_location.iteritems():
            for node2, latlon2 in node_location.iteritems():
                lat1, lon1 = latlon1
                lat2, lon2 = latlon2
                dd = distance(lat1, lon1, lat2, lon2)
                if dd > max_acceptable_distance:
                    if ((node1, node2) in edges or (node2, node1) in edges):
                        try:
                            params.mention_graph.remove_edge(node1, node2)
                            num_nodes_removed += 1
                        except:
                            pass
        print(str(num_nodes_removed) + ' edges removed from the graph') 
                
    use_shortest_paths = False
    shortest_paths = {}
    if use_shortest_paths:
        shortest_paths = nx.all_pairs_shortest_path_length(params.mention_graph, cutoff=3)
    
    
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
            nbrs = params.mention_graph[node]
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
                    edge_weight = params.mention_graph[node][nbr]['weight']
                    nbrlats.append(lat)
                    nbrlons.append(lon)
                    if normalize_edge:
                        edge_weight_normalized = float(edge_weight * edge_weight) / (params.mention_graph.degree(nbr) * params.mention_graph.degree(node))
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
            user_degree = params.mention_graph.degree([node_id[u] for u in evalUsers.keys()])
            for u in evalUsers:
                evalUserId = node_id[u]
                degree = user_degree[evalUserId]
                degrees.append(degree)
    
    return current_mean, current_median, current_acc

def LP_classification(weighted=True, normalize_edge=False, remove_celebrities=False, dev=False, project_to_main_users=False, node_order='l2h', remove_mentions_with_degree_one=True, negative_sampling=False):
    '''
    This function implements iterative label propagation as in Modified Adsorption without the regulariser term.
    This is not parallel and is slower than running Junto.
    The labels are the result of running k-d tree over the training coordinates which discretises
    the real-valued coordinates into several regions with different area but the same number of users.
    The label distribution of training users are kept unchanged. The label distribution of test/dev users
    are updated to the mean of their neighbours.
    If project_to_main_users is True the network will be collapsed (keeping just training and test/dev users) and
    if False, the complete @-mention graph will be used.
    Note: The results reported in the paper are not based on this function. They are based on label propagation
    using Modified Adsorption using Junto implementation (https://github.com/parthatalukdar/junto).
    '''
    params.U_train = [u for u in sorted(params.trainUsers)]
    params.U_test = [u for u in sorted(params.testUsers)]
    params.U_dev = [u for u in sorted(params.devUsers)]
    num_classes = len(params.categories)
    logging.info('running classification based label propagation') 
    U_eval = []
    if dev:
        evalText = params.devText
        evalUsers = params.devUsers
        U_eval = params.U_dev
        eval_classes = params.devClasses
    else:
        evalText = params.testText
        evalUsers = params.testUsers
        U_eval = params.U_test
        eval_classes = params.testClasses

    U_all = params.U_train + U_eval
    assert len(U_all) == len(params.U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    id_node = dict(zip(idx, U_all))


    
    params.mention_graph = nx.Graph()
    
    print('weighted=%s and params.prior=%s' % (weighted, params.prior))

    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    l = len(params.trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    params.mention_graph.add_nodes_from(node_id.values())
    for user, text in params.trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id] 
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        user_id = node_id[user]
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id]
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)  
        
    
    trainuserid_location = {}
    evaluserid_location = {}
    trainLats = []
    trainLons = []
    node_location = {}
    dongle_nodes = []
    text_preds = {}
    
    for user, loc in params.trainUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)
        trainuserid_location[user_id] = (lat, lon)
        
    for user, loc in evalUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        evaluserid_location[user_id] = (lat, lon)
    
    print "the number of train nodes is " + str(len(params.trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    median_lat = np.median(trainLats)
    median_lon = np.median(trainLons)
    median_classIndx, dist = assignClass(median_lat, median_lon)
    celebrity_threshold = params.celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = params.mention_graph.nodes_iter()
        for node in nodes:
            nbrs = params.mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in node_id.values():
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
                params.mention_graph.remove_node(celebrity)
    





    if remove_mentions_with_degree_one:
        mention_nodes = set(params.mention_graph.nodes()) - set(node_id.values())
        mention_degree = params.mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = {node for node, degree in mention_degree.iteritems() if degree < 2}
        logging.info('found ' + str(len(one_degree_non_target)) + ' params.mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            params.mention_graph.remove_node(node)
            
    if project_to_main_users:
        logging.info('projecting the graph into the target user.')
        main_users = node_id.values()
        # params.mention_graph = bipartite.overlap_weighted_projected_graph(params.mention_graph, main_users, jaccard=False)
        params.mention_graph = efficient_collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str=None, degree_power=1, caller='lp')
        # params.mention_graph = collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str='weight')
        # params.mention_graph = bipartite.projected_graph(params.mention_graph, main_users)
    logging.info("Edge number: " + str(params.mention_graph.number_of_edges()))
    logging.info("Node number: " + str(params.mention_graph.number_of_nodes()))

    node_labeldist = {}
    logging.info('initialising user label distributions...')
    for node, id in node_id.iteritems():
        if id < len(params.U_train):
            label = params.trainClasses[node]
            dist = lil_matrix((1, len(params.categories)))
            dist[0, label] = 1
            node_labeldist[id] = dist
        
    
    # check the effect of geolocation order on performance. The intuition is that
    # if the high confident nodes are geolocated first it may be better because noisy
    # predictions won't propagate from the first iterations.
    if node_order:
        node_degree = params.mention_graph.degree()
        if node_order == 'h2l':
            reverse_order = True
        else:
            reverse_order = False
        nodes = sorted(node_degree, key=node_degree.get, reverse=reverse_order)
        if node_order == 'random':
            random.shuffle(nodes)
        # nodes_unknown = [node for node in params.mention_graph.nodes() if node not in trainUsersLower and node not in dongle_nodes]
        nodes_unknown = [node for node in nodes if node not in trainuserid_location and node not in dongle_nodes]
    else:
        nodes_unknown = [node for node in params.mention_graph.nodes() if node not in trainuserid_location and node not in dongle_nodes]

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
            neighbors_labeldist = lil_matrix((1, len(params.categories)))
            nbrs = params.mention_graph[node]    
            sum_nbr_edge_weights = 0.0
            for nbr, edge_data in nbrs.iteritems():
                if nbr in node_labeldist:
                    should_be_updated = True
                    nbrlabeldist = node_labeldist[nbr]
                    edge_weight = edge_data['weight']
                    sum_nbr_edge_weights += edge_weight
                    neighbors_labeldist = neighbors_labeldist + edge_weight * nbrlabeldist
            
            negative_labeldist = lil_matrix((1, len(params.categories)))
            sum_negative_edge_weights = 0
            if negative_sampling:
                #select #nbrs random nodes from the graph
                num_negative = max(len(nbrs) / 2, 1)
                negative_nodes = random.sample(params.mention_graph.nodes(), num_negative)
                for neg_n in negative_nodes:
                    if neg_n in node_labeldist:
                        negative_labeldist += node_labeldist[neg_n]
                        sum_negative_edge_weights += 1
                
            if should_be_updated:
                old_labeldist = node_labeldist.get(node, csr_matrix((1, len(params.categories))))
                new_labeldist = (sum_nbr_edge_weights + sum_negative_edge_weights) * old_labeldist + neighbors_labeldist - negative_labeldist
                new_labeldist[new_labeldist < 0] = 0
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
        loss(preds=eval_predictions, U_eval=U_eval) 
        iter_num += 1
        if iter_num == max_iter:
            converged = True
        
        if len(node_labeldist) == located_nodes_count:
            logging.info("converged. No new nodes added in this iteration.")
            # converged = True
        
    
def LP_classification_edgexplain(weighted=True, prior='none', normalize_edge=False, remove_celebrities=False, dev=False, project_to_main_users=False, node_order='l2h', remove_mentions_with_degree_one=True, add_text=False):
    '''
    This function implements label propagation using the edgexplain idea over discretized regions
    for geolocation of social media users.
    '''
    params.U_train = [u for u in sorted(params.trainUsers)]
    params.U_test = [u for u in sorted(params.testUsers)]
    params.U_dev = [u for u in sorted(params.devUsers)]
    num_classes = len(params.categories)
    logging.info('running classification based label propagation with edgexplain') 
    U_eval = []
    if dev:
        evalText = params.devText
        evalUsers = params.devUsers
        U_eval = params.U_dev
        eval_classes = params.devClasses
    else:
        evalText = params.testText
        evalUsers = params.testUsers
        U_eval = params.U_test
        eval_classes = params.testClasses

    U_all = params.U_train + U_eval
    assert len(U_all) == len(params.U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    

    
    params.mention_graph = nx.Graph()
    
    print('weighted=%s and params.prior=%s' % (weighted, prior))

    print "building the direct graph"
    token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern1 = re.compile(token_pattern1)
    token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern2 = re.compile(token_pattern2)
    l = len(params.trainText)
    tenpercent = l / 10
    i = 1
    # add train and test users to the graph
    params.mention_graph.add_nodes_from(node_id.values())
    for user, text in params.trainText.iteritems():
        user_id = node_id[user]    
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1  
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id] 
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)   
       
    print "adding the eval graph"
    for user, text in evalText.iteritems():
        user_id = node_id[user]
        params.mentions = [node_id.get(u.lower(), u.lower()) for u in token_pattern1.findall(text)]
        params.mentions = [m for m in params.mentions if m != user_id]
        mentionDic = Counter(params.mentions)
        params.mention_graph.add_nodes_from(mentionDic.keys())
        for mention, freq in mentionDic.iteritems():
            if not weighted:
                freq = 1
            if params.mention_graph.has_edge(user_id, mention):
                params.mention_graph[user_id][mention]['weight'] += freq
                # params.mention_graph[mention][user]['weight'] += freq/2.0
            else:
                params.mention_graph.add_edge(user_id, mention, weight=freq)
                # params.mention_graph.add_edge(mention, user, weight=freq/2.0)  
        

    evaluserid_location = {}
    trainLats = []
    trainLons = []

    
    for user, loc in params.trainUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)

        
    for user, loc in evalUsers.iteritems():
        user_id = node_id[user]
        lat, lon = locationStr2Float(loc)
        evaluserid_location[user_id] = (lat, lon)
    
    print "the number of train nodes is " + str(len(params.trainUsers))
    print "the number of test nodes is " + str(len(evalUsers))
    median_lat = np.median(trainLats)
    median_lon = np.median(trainLons)
    median_classIndx, dist = assignClass(median_lat, median_lon)
    celebrity_threshold = params.celeb_threshold
    celebrities = []
    if remove_celebrities:
        nodes = params.mention_graph.nodes_iter()
        for node in nodes:
            nbrs = params.mention_graph.neighbors(node)
            if len(nbrs) > celebrity_threshold:
                if node not in node_id.values():
                    celebrities.append(node)
        print("found %d celebrities with celebrity threshold %d" % (len(celebrities), celebrity_threshold))
        for celebrity in celebrities:
                params.mention_graph.remove_node(celebrity)
    





    if remove_mentions_with_degree_one:
        mention_nodes = set(params.mention_graph.nodes()) - set(node_id.values())
        mention_degree = params.mention_graph.degree(nbunch=mention_nodes, weight=None)
        one_degree_non_target = {node for node, degree in mention_degree.iteritems() if degree < 2}
        logging.info('found ' + str(len(one_degree_non_target)) + ' params.mentions with degree 1 in the graph.')
        for node in one_degree_non_target:
            params.mention_graph.remove_node(node)
            
    if project_to_main_users:
        logging.info('projecting the graph into the target user.')
        main_users = node_id.values()
        # params.mention_graph = bipartite.overlap_weighted_projected_graph(params.mention_graph, main_users, jaccard=False)
        params.mention_graph = efficient_collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str=None, degree_power=1, caller='lp')
        # params.mention_graph = collaboration_weighted_projected_graph(params.mention_graph, main_users, weight_str='weight')
        # params.mention_graph = bipartite.projected_graph(params.mention_graph, main_users)
    logging.info("Edge number: " + str(params.mention_graph.number_of_edges()))
    logging.info("Node number: " + str(params.mention_graph.number_of_nodes()))
    #save adjacancy matrix for future use
    non_target_nodes = sorted(list(set(params.mention_graph.nodes()) - set(idx)))
    if len(non_target_nodes) == 0:
        non_target_node_indices = []
    else:
        non_target_node_indices = range(len(U_all), params.mention_graph.number_of_nodes())
    A = nx.adjacency_matrix(params.mention_graph, nodelist=idx + non_target_nodes, weight='weight')
    

    
    num_nodes = params.mention_graph.number_of_nodes()
    X = lil_matrix((num_nodes, len(params.categories)))
    training_indices = []
    test_indices = []
    for u in params.U_train:
        id = node_id[u]
        label = params.trainClasses[u]
        X[id, label] = 1
        training_indices.append(id)
    for u in U_eval:
        id = node_id[u]
        test_indices.append(id)

    X=X.toarray()
    if add_text:
        do_lda = False
        #scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
        scaler = StandardScaler(copy=False)
        text_dimensions = None
        train_text = [params.trainText[u] for u in params.U_train]
        eval_text = [evalText[u] for u in U_eval]
        params.X_train, X_eval, vocab = utils.vectorize(train_text, eval_text, params.data_encoding, binary= not do_lda)
        print '#features:', params.X_train.shape[1]
        
        if do_lda:
            params.X_train, X_eval= utils.topicModel(params.X_train, X_eval, components=text_dimensions, vocab=vocab)
            X_text = np.vstack((params.X_train, X_eval))
            pairwise_similarity = np.dot(X_text, np.transpose(X_text))
        else:
            X_text = sparse.vstack([params.X_train, X_eval], format='csr')
            pairwise_similarity = X_text.dot(X_text.transpose(copy=True))
        C = scaler.fit_transform(pairwise_similarity.toarray())
        np.fill_diagonal(C, 0)
        #C = normalize(C, norm='l1', axis=1, copy=False)
            
    else:
        text_dimensions = None
        C = np.empty((num_nodes, num_nodes))
        C.fill(0)
    
    optimise_with_sgd = False
    if optimise_with_sgd:
        training_indices = np.array(training_indices)
        for learning_rate in [0.1]:
            for alpha in [10]:
                    for lambda1 in [0.5]:                    
                        X_new = edgexplain.edgexplain_geolocate(X, training_indices, A, iterations=20, learning_rate=learning_rate, alpha=alpha, C=C, lambda1=lambda1, text_dimensions=text_dimensions)
                        X_new = X_new[:, 0:len(params.categories)]
                        preds = np.argmax(X_new, axis=1)
                        train_preds = preds[training_indices]
                        test_preds = preds[test_indices]
                        print 'train error'
                        loss(preds=train_preds.tolist(), U_eval=params.U_train)
                        print 'learning rate', learning_rate, 'alpha', alpha, 'lambda1', lambda1
                        print 'eval error'
                        loss(preds=test_preds.tolist(), U_eval=U_eval)
    else:                
        X_new = edgexplain.edgexplain_geolocate_iterative(X=lil_matrix(X), G=params.mention_graph,train_ids = training_indices, test_ids=test_indices + non_target_nodes, id_index=dict(zip(idx + non_target_nodes, idx + non_target_node_indices)), label_slices=[[0, len(params.categories)]], preserve_coef=0.9, iterations=10, alpha=1, C=0, node_order='random', keep_topK=10, edgexplain__scaler=True)
        X_new = X_new[:, 0:len(params.categories)].toarray()
        preds = np.argmax(X_new, axis=1)
        train_preds = preds[training_indices]
        test_preds = preds[test_indices]
        print 'train error'
        loss(preds=train_preds.tolist(), U_eval=params.U_train)
        print 'eval error'
        loss(preds=test_preds.tolist(), U_eval=U_eval)


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
            #if G.has_edge(n1, n2):
                #weight += G[n1][n2]['weight']
                #pass 
            vnbrs = set(pred[n2])
            common = unbrs & vnbrs
            #all = unbrs | vnbrs

            if weight_str is not None:
                weight += sum([1.0 * B[n1][n][weight_str] * B[n2][n][weight_str] / ((len(B[n]) - 1) ** degree_power) for n in common if len(B[n]) > 1])
            else:
                weight += sum([1.0 / ((len(B[n]) - 1) ** degree_power) for n in common if len(B[n]) > 1])
                #weight += float(len(common)) / len(all)
            if weight != 0:
                G.add_edge(n1, n2, weight=weight)
    
    
    return G

def efficient_collaboration_weighted_projected_graph(B, nodes, weight_str=None, degree_power=1, caller='lp'):
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
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    i = 0
    tenpercent = len(all_nodes) / 10
    for m in all_nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  

        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]
        if len(nbrs) < 2:
            continue
        if m in nodes:
            for n in target_nbrs:
                if m < n:
                    n_nbrs = len(B[n])
                    if n_nbrs > 1:
                        w_n = 1.0 / (n_nbrs - 1)
                    else:
                        w_n = 0
                    w = 1.0 / (len(nbrs) - 1) + w_n
                    if G.has_edge(m, n):
                        G[m][n]['weight'] += w
                    else:
                        G.add_edge(m, n, weight=w)
        for n1 in target_nbrs:
            for n2 in target_nbrs:
                if n1 < n2:
                    w = 1.0 / (len(nbrs) - 1)
                    if G.has_edge(n1, n2):
                        G[n1][n2]['weight'] += w
                    else:
                        G.add_edge(n1, n2, weight=w)
    return G

def junto_postprocessing(multiple=False, dev=False, method='median', celeb_threshold=5, weighted=False, text_prior='none', postfix=''):
    EVALUATE_REAL_VALUED = False

    lats = params.classLatMedian.values()
    lons = params.classLonMedian.values()
    median_lat = np.median(lats)
    median_lon = np.median(lons)
    classIndx, dist = assignClass(median_lat, median_lon)
    trainUsersLower = [u.lower() for u in sorted(params.trainUsers)]
    
    
    params.U_test = [u for u in sorted(params.testUsers)]
    params.U_dev = [u for u in sorted(params.devUsers)]
    if dev:
        U_eval = params.U_dev
        devStr = '.dev'
    else:
        U_eval = params.U_test
        devStr = ''
    result_dump_file = path.join(params.GEOTEXT_HOME, 'results-' + params.DATASETS[params.DATASET_NUMBER - 1] + '-' + method + '-' + str(params.BUCKET_SIZE) + '-' + params.textmodel + '.pkl')
    if params.prior != 'none':
        logging.info("reading (preds, devPreds, params.U_test, params.U_dev, testProbs, devProbs) from " + result_dump_file)
        with open(result_dump_file, 'rb') as inf:
            preds_text, devPreds_text, U_test_text, U_dev_text, testProbs_text, devProbs_text = pickle.load(inf)
            logging.info("text test results:")
            loss(preds_text, U_test_text)
            logging.info("text dev results:")
            loss(devPreds_text, U_dev_text)
            
            if dev:
                text_preds = devPreds_text
                text_probs = devProbs_text
                assert params.U_dev == U_dev_text, "text users are not equal to loaded dev users"
                
            else:
                text_preds = preds_text
                text_probs = testProbs_text
                assert params.U_test == U_test_text, "text users are not equal to loaded test users"
    
    # split a training set and a test set
    params.Y_test = np.asarray([params.testClasses[u] for u in params.U_test])
    params.Y_dev = np.asarray([params.devClasses[u] for u in params.U_dev])
    
    textStr = '.' + params.prior
    if params.prior == 'none':
        textStr = ''
    weightedStr = '.weighted'
    if not weighted:
        weightedStr = ''
    

        
    if not multiple:
        
        files = [path.join(params.GEOTEXT_HOME, 'label_prop_output_' + params.DATASETS[params.DATASET_NUMBER - 1] + '_' + method + '_' + str(params.BUCKET_SIZE) + '_' + str(params.celeb_threshold) + devStr + textStr + weightedStr + postfix)]
    else:
        junto_output_dir = '/home/arahimi/git/junto-master/examples/simple/data/outputs_' + params.DATASETS[params.DATASET_NUMBER - 1]
        files = glob.glob(junto_output_dir + '/label_prop_output*')
        files = sorted(files)
    # feature_extractor2(min_df=50)

    for junto_output_file in files:   
        id_name_file = path.join(params.GEOTEXT_HOME, 'id_user' + '_' + method + '_' + str(params.BUCKET_SIZE) + devStr + textStr + weightedStr) 
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
                        u = params.U_test[int(uid) - len(params.trainUsers)]
                        
                        lat2, lon2 = locationStr2Float(params.userLocation[u])
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
                            labelProb = float(label_scores.split()[1])
                            if label == '__DUMMY__':
                                # pass
                                logging.info('choosing second ranked label as the first one is a dummy!')
                                label = label_scores.split()[2]
                                labelProb = float(label_scores.split()[2])
                                second_option_selected += 1
                        except:
                            print fields
                        
                        # Tracer()()
                        if label == '__DUMMY__':
                            dummy_count += 1
                            if params.prior == 'backoff':
                                text_predition = str(text_preds[user_index])
                                label = text_predition
                            else:
                                # label = str(len(params.categories) / 2)
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
                        if params.prior == 'backoff':
                            text_predition = str(text_preds[int(uid) - len(trainUsersLower)])
                            pred = text_predition
                        else:
                            # pred = str(len(params.categories) / 2)  
                            pred = classIndx
                        name_pred[u] = pred

                    preds.append(int(pred))
                
                print "doubles found is " + str(doubles_found)
                print "users with dummy labels: " + str(dummy_count)
                print "users not in network: " + str(user_not_in_network)
                print "total number of users: " + str(len(U_eval))
                # print preds
                # print [int(i) for i in params.Y_test.tolist()]    
                mad_mean, mad_median, mad_acc = loss(preds, U_eval)
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

def run_planteoid():
    params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names = feature_extractor(norm=params.norm, use_mention_dictionary=None, min_df=10, max_df=0.2, stop_words='english', binary=params.binary, sublinear_tf=params.sublinear, vocab=None, use_idf=params.use_idf, save_vectorizer=False)
    LP_collapsed(weighted=False, normalize_edge=True, remove_celebrities=True, dev=True, project_to_main_users=True, node_order='random', remove_mentions_with_degree_one=True)
    sys.path.append(path.abspath('../planetoid'))
    import argparse
    from scipy import sparse
    import numpy as np
    import random
    from collections import defaultdict as dd
    import model
    """set the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', help = 'learning rate', type = float, default = 0.1)
    parser.add_argument('--neg_samp', help = 'negative sampling rate', type = int, default = 5)
    parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
    parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
    parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
    parser.add_argument('--batch_size', help = 'the size of batch for training instances', type = int, default = 200)
    parser.add_argument('--g_batch_size', help = 'the batch size for graph', type = int, default = 200)
    parser.add_argument('--g_sample_size', help = 'the sample size from label information', type = int, default = 100)
    parser.add_argument('--g_learning_rate', help = "the learning rate of graphs", type = float, default = 1e-2)
    args = parser.parse_args()
    
    Y_train = np.zeros((len(params.trainUsers), len(params.categories)), dtype = np.int32)
    Y_dev = np.zeros((len(params.devUsers), len(params.categories)), dtype = np.int32)
    Y_test = np.zeros((len(params.testUsers), len(params.categories)), dtype = np.int32)
    for i in range(len(params.trainUsers)):
        Y_train[i, params.Y_train[i]] = 1
    for i in range(len(params.devUsers)):
        Y_dev[i, params.Y_dev[i]] = 1
    for i in range(len(params.testUsers)):
        Y_test[i, params.Y_test[i]] = 1
    
    U_eval = params.U_dev
    U_all = params.U_train + U_eval
    assert len(U_all) == len(params.U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    planteoid_graph = defaultdict(list)
    for node, id in node_id.iteritems():
        planteoid_graph[id] = params.mention_graph[id].keys()
    x = csr_matrix(params.X_train, dtype=np.float32)
    y = Y_train
    tx = csr_matrix(params.X_dev, dtype=np.float32)
    ty = Y_dev
    allx = sparse.vstack([x, tx])
    ally = np.vstack((y, ty))
    graph = planteoid_graph
    test_indices = list(range(x.shape[0], allx.shape[0]))
    dataset_name = params.DATASETS[params.DATASET_NUMBER - 1]
    planteoid_data = '/home/arahimi/git/planetoid2/planetoid/data'
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.allx'), 'wb') as fout:
        pickle.dump(allx, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.x'), 'wb') as fout:
        pickle.dump(x, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.tx'), 'wb') as fout:
        pickle.dump(tx, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.ally'), 'wb') as fout:
        pickle.dump(ally, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.y'), 'wb') as fout:
        pickle.dump(y, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.ty'), 'wb') as fout:
        pickle.dump(ty, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.graph'), 'wb') as fout:
        pickle.dump(graph, fout)
    with open(path.join(planteoid_data, 'ind.' + dataset_name + '.test.index'), 'w') as fout:
        for _ in test_indices:
            fout.write(str(_) + '\n')
    
    

    
    pdb.set_trace()
    """build the model and start training"""
    m = model.model(args)
    m.add_data(csr_matrix(params.X_train, dtype=np.float32), csr_matrix(params.X_dev, dtype=np.float32), Y_train, Y_dev, planteoid_graph)
    m.build()
    m.joint_train()
    preds = np.argmax(m.predictions, axis=1)
    loss(preds, U_eval=params.U_dev)
    pdb.set_trace()

def run_planetoid():
    params.X_train, params.Y_train, params.U_train, params.X_dev, params.Y_dev, params.U_dev, params.X_test, params.Y_test, params.U_test, params.categories, params.feature_names = feature_extractor(norm=params.norm, use_mention_dictionary=None, min_df=10, max_df=0.2, stop_words='english', binary=params.binary, sublinear_tf=params.sublinear, vocab=None, use_idf=params.use_idf, save_vectorizer=False)
    LP_collapsed(weighted=False, normalize_edge=True, remove_celebrities=True, dev=True, project_to_main_users=True, node_order='random', remove_mentions_with_degree_one=True)
    sys.path.append(path.abspath('../planetoid2/planetoid'))
    import argparse
    from scipy import sparse
    import numpy as np
    import random
    from collections import defaultdict as dd
    from ind_model import ind_model as model
    """set the arguments"""
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.1)
    parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = len(params.categories) * 4)
    parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 2)
    parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
    parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 1000)
    parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 1000)
    parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 100)
    parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 0)
    parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1e-3)
    parser.add_argument('--model_file', help = 'filename for saving models', type = str, default = 'ind.model')
    parser.add_argument('--use_feature', help = 'whether use input features', type = bool, default = True)
    parser.add_argument('--update_emb', help = 'whether update embedding when optimizing supervised loss', type = bool, default = True)
    parser.add_argument('--layer_loss', help = 'whether incur loss on hidden layers', type = bool, default = True)
    args = parser.parse_args()
    
    Y_train = np.zeros((len(params.trainUsers), len(params.categories)), dtype = np.int32)
    Y_dev = np.zeros((len(params.devUsers), len(params.categories)), dtype = np.int32)
    Y_test = np.zeros((len(params.testUsers), len(params.categories)), dtype = np.int32)
    for i in range(len(params.trainUsers)):
        Y_train[i, params.Y_train[i]] = 1
    for i in range(len(params.devUsers)):
        Y_dev[i, params.Y_dev[i]] = 1
    for i in range(len(params.testUsers)):
        Y_test[i, params.Y_test[i]] = 1
    
    U_eval = params.U_dev
    U_all = params.U_train + U_eval
    assert len(U_all) == len(params.U_train) + len(U_eval), "duplicate user problem"
    idx = range(len(U_all))
    node_id = dict(zip(U_all, idx))
    planteoid_graph = defaultdict(list)
    for node, id in node_id.iteritems():
        planteoid_graph[id] = params.mention_graph[id].keys()
    x = csr_matrix(params.X_train, dtype=np.float32)
    y = Y_train
    tx = csr_matrix(params.X_dev, dtype=np.float32)
    ty = Y_dev
    allx = sparse.vstack([x, tx])
    ally = np.vstack((y, ty))
    graph = planteoid_graph
    test_indices = list(range(x.shape[0], allx.shape[0]))
    dataset_name = params.DATASETS[params.DATASET_NUMBER - 1]
    write_to_pickle = False
    if write_to_pickle:
        planteoid_data = '/home/arahimi/git/planetoid2/planetoid/data'
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.allx'), 'wb') as fout:
            pickle.dump(allx, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.x'), 'wb') as fout:
            pickle.dump(x, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.tx'), 'wb') as fout:
            pickle.dump(tx, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.ally'), 'wb') as fout:
            pickle.dump(ally, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.y'), 'wb') as fout:
            pickle.dump(y, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.ty'), 'wb') as fout:
            pickle.dump(ty, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.graph'), 'wb') as fout:
            pickle.dump(graph, fout)
        with open(path.join(planteoid_data, 'ind.' + dataset_name + '.test.index'), 'w') as fout:
            for _ in test_indices:
                fout.write(str(_) + '\n')
    
    

    def comp_accu(tpy, ty):
        return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]
    
    m = model(args)                                                 # initialize the model
    m.add_data(x, y, allx, graph)                                   # add data
    m.build()                                                       # build the model
    m.init_train(init_iter_label = 10000, init_iter_graph = 400)    # pre-training
    iter_cnt, max_accu = 0, 0
    max_mean, max_median, max_acc = 0, 0, 0
    while True:
        m.step_train(max_iter = 1, iter_graph = 0.1, iter_inst = 1, iter_label = 0) # perform a training step
        tpy = m.predict(tx)            # predict the dev set
        preds = np.argmax(tpy, axis = 1)                    
        _mean, _median, _acc = loss(preds, U_eval=params.U_dev)
        if _acc > max_acc:
            max_acc = _acc
            max_mean = _mean
            max_median = _median                     
        accu = comp_accu(tpy, ty)                                                   # compute the accuracy on the dev set
        print iter_cnt, accu, max_accu
        iter_cnt += 1
        if accu > max_accu:
            m.store_params()                                                        # store the model if better result is obtained
            max_accu = max(max_accu, accu)
        logging.info(str(max_mean) +' ' +  str(max_median) + ' ' + str(max_acc))
    
if __name__ == '__main__':

    initialize(granularity=params.BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=params.do_not_discretize)
    #prepare_adsorption_data_collapsed(DEVELOPMENT=False, text_prior=params.prior, CELEBRITY_THRESHOLD=params.celeb_threshold, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=False)
    
    #junto_postprocessing(multiple=False, dev=False, method='median', celeb_threshold=params.celeb_threshold, weighted=False, text_prior=params.prior, postfix='')
    #sys.exit()
    #run_planteoid()
    #run_planetoid()
    if 'text_classification' in params.models_to_run:
        t_mean, t_median, t_acc, d_mean, d_median, d_acc = asclassification(granularity=params.BUCKET_SIZE, use_mention_dictionary=False, binary=params.binary, sublinear=params.sublinear, penalty=params.penalty, fit_intercept=params.fit_intercept, norm=params.norm, use_idf=params.use_idf)
    pdb.set_trace()    
    if 'network_lp_regression_collapsed' in params.models_to_run:
        LP_collapsed(weighted=False, normalize_edge=True, remove_celebrities=True, dev=False, project_to_main_users=False, node_order='random', remove_mentions_with_degree_one=False)
    if 'network_lp_regression' in params.models_to_run:
        LP(weighted=False, normalize_edge=True, remove_celebrities=True, dev=False, node_order='random')
    if 'network_lp_classification' in params.models_to_run:
        LP_classification(weighted=True, normalize_edge=False, remove_celebrities=False, dev=True, project_to_main_users=True, node_order='random', remove_mentions_with_degree_one=True)
    if 'network_lp_classification_edgexplain' in params.models_to_run:
        LP_classification_edgexplain(weighted=True, normalize_edge=False, remove_celebrities=False, dev=True, project_to_main_users=False, node_order='random', remove_mentions_with_degree_one=True)
    

    # junto_postprocessing(multiple=False, dev=False, text_confidence=1.0, method=params.partitionMethod, params.celeb_threshold=5, weighted=True, text_prior=True)

    print str(datetime.now())
    script_end_time = time.time()
    script_execution_hour = (script_end_time - script_start_time) / 3600.0
    print "The script execution time (in hours) is " + str(script_execution_hour)
