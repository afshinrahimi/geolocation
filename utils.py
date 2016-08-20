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
import params
import pdb
import geolocate
def fusion_table(Y_train=None, filename='fusion.csv'):
    colors = ['ranger_station','factory','gas_stations','post_office','agriculture','funicular','post_office_jp','thunderstorm','traffic','polygon','cross_hairs','poi','falling_rocks','police_badge','terrain','capital_small','placemark_square','square','cross_hairs_highlight','ruler','donut','play','stable','capital_small_highlight','placemark_square_highlight','stop','road_shield3','star','shaded_dot','wht_pushpin','dining','pharmacy_rx','sledding','capital_big','placemark_circle','pause','road_shield2','triangle','target','info_i','crosscountry_ski','pharmacy_plus','ski_lift','capital_big_highlight','placemark_circle_highlight','go','road_shield1','open_diamond','arrow_reverse','forbidden','convenience','parks','shower','coffee','parking_lot','sea_ports','1_blue','2_blue','3_blue','4_blue','5_blue','6_blue','7_blue','8_blue','9_blue','10_blue','cemetary','museum','schools','a_blue','b_blue','c_blue','d_blue','e_blue','f_blue','g_blue','h_blue','i_blue','j_blue','k_blue','l_blue','m_blue','n_blue','o_blue','p_blue','q_blue','r_blue','s_blue','t_blue','u_blue','v_blue','w_blue','x_blue','y_blue','z_blue','cemetary_jp','mountains','rec_wheel_chair_accessible','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','car_ferry','lighthouse','rec_phone','canoeing','library','rec_parking_lot','blu_stars','grn_stars','ltblu_stars','orange_stars','pink_stars','purple_stars','red_stars','wht_stars','ylw_stars','buildings','landmark','rec_lodging','prayer','blu_square','grn_square','ltblu_square','orange_square','pink_square','purple_square','red_square','wht_square','ylw_square','broken_link','highway','rec_info_circle','temple_jp','blu_diamond','grn_diamond','ltblu_diamond','orange_diamond','pink_diamond','purple_diamond','red_diamond','wht_diamond','ylw_diamond','boat_launch','heliport','rec_gas_stations','synagogue','blu_circle','grn_circle','ltblu_circle','orange_circle','pink_circle','purple_circle','red_circle','wht_circle','ylw_circle','binoculars','grocery','rec_dining','shrine_jp','blu_blank','grn_blank','ltblu_blank','orange_blank','pink_blank','purple_blank','red_blank','wht_blank','ylw_blank','bars','govtbldgs','rec_convenience','mosque','arena','gondola','rec_bus','hindu_temple','large_red','large_yellow','large_green','large_blue','large_purple','airports','golf','ranger_tower','church','small_red','small_yellow','small_green','small_blue','small_purple','measle_brown','measle_grey','measle_white','measle_turquoise','arrow','arts','campfire','cycling','flag','bus','yen','earthquake','firedept','homegardenbusiness','cabs','euro','camera','caution','campground','dollar','electronics','hiker','fishing','horsebackriding','hospitals','info','info_circle','lodging','man','marina','mechanic','motorcycling','movies','partly_cloudy','phone','picnic','police','rail','rainy','realestate','sailing','salon','shopping','ski','snack_bar','snowflake_simple','subway','sunny','swimming','toilets','trail','tram','truck','volcano','water','webcam','wheel_chair_accessible','woman','geographic_features']
    with open(filename, 'w') as outf:
        for i, u in enumerate(params.U_train):
            latlon = params.trainUsers[u]
            if Y_train is None:
                lbl = params.trainClasses[u]
            else:
                lbl = Y_train[i]
            lbl = colors[int(lbl)]
            outf.write(latlon + ',' + str(lbl) + ',' + u + '\n')
    
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
def tsne1():
    from sklearn.decomposition import PCA
    from collections import defaultdict
    pca = PCA(n_components=2)
    with open('emnlp2016/word_emb_words.pkl', 'rb') as inf:
        w_emb, ws = pickle.load(inf)
    index_w = dict(zip(range(len(ws)), ws))
    w_index = dict(zip(ws, range(len(ws))))
    cities_50 = ['nyc','la','Chicago','Houston','Philadelphia','Phoenix','San Antonio','San Diego','Dallas','San Jose','Austin','Jacksonville','San Francisco','Indianapolis','Columbus','Fort Worth','Charlotte','Seattle','Denver','El Paso','Detroit','Washington','Boston','Memphis','Nashville','Portland','Oklahoma','Las Vegas','Baltimore','Louisville','Milwaukee','Albuquerque','Tucson','Fresno','Sacramento','Kansas City','Long Beach','Mesa','Atlanta','Colorado Springs','Virginia Beach','Raleigh','Omaha','Miami','Oakland','Minneapolis','Tulsa','Wichita','New Orleans','Arlington']
    states_50 = ['California','Texas','Florida','New York','Illinois','Pennsylvania','Ohio','Georgia','North Carolina','Michigan','New Jersey','Virginia','Washington','Arizona','Massachusetts','Indiana','Tennessee','Missouri','Maryland','Wisconsin','Minnesota','Colorado','South Carolina','Alabama','Louisiana','Kentucky','Oregon','Oklahoma','Connecticut','Iowa','Utah','Mississippi','Arkansas','Kansas','Nevada','New Mexico','Nebraska','West Virginia','Idaho','Hawaii','New Hampshire','Maine','Rhode Island','Montana','Delaware','South Dakota','North Dakota','Alaska','Vermont','Wyoming']
    #all_cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Philadelphia', 'Phoenix', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Indianapolis', 'San Francisco', 'Columbus', 'Fort Worth', 'Charlotte', 'Detroit', 'El Paso', 'Memphis', 'Boston', 'Boston', 'Denver', 'Washington', 'Nashville', 'Baltimore', 'Louisville', 'Portland', 'Oklahoma', 'Milwaukee', 'Las Vagas', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Long Beach', 'Kansas City', 'Mesa', 'Virginia Beach', 'Atlanta', 'Colorado Springs', 'Raleigh', 'Omaha', 'Miami', 'Oakland', 'Tulsa', 'Omaha', 'Cleveland', 'Omaha', 'Arlington', 'Omaha', 'Bakersfield', 'Tampa', 'Honolulu', 'Anaheim', 'Aurora', 'Santa Ana', 'St. Louis', 'Riverside', 'Corpus Christi', 'Pittsburgh', 'Lexington', 'Anchorage', 'Stockton', 'Cincinnati', 'Saint Paul', 'Toledo', 'Newark', 'Greensboro', 'Plano', 'Henderson', 'Lincoln', 'Buffalo', 'Fort Wayne', 'Jersey City', 'Chula Vista', 'Orlando', 'St. Petersburg', 'Norfolk', 'Chandler', 'Laredo', 'Madison', 'Durham', 'Lubbock', 'Winston Salem', 'Garland', 'Glendale', 'Hialeah', 'Reno', 'Baton Rouge', 'Irvine', 'Chesapeake', 'Irving', 'Scottsdale', 'North Las Vegas', 'Fremont', 'Gilbert', 'San Bernardino', 'Boise[', 'Birmingham', 'Rochester', 'Richmond', 'Spokane', 'Des Moines', 'Montgomery', 'Modesto', 'Fayetteville', 'Tacoma', 'Shreveport', 'Fontana', 'Oxnard', 'Aurora', 'Moreno Valley', 'Akron', 'Yonkers', 'Columbus', 'Augusta', 'Little Rock', 'Amarillo', 'Mobile', 'Huntington Beach', 'Glendale', 'Grand Rapids', 'Salt Lake City', 'Tallahassee', 'Huntsville', 'Worcester', 'Knoxville', 'Grand Prairie', 'Newport News', 'Brownsville', 'Santa Clarita', 'Overland Park', 'Providence', 'Jackson', 'Garden Grove', 'Oceanside', 'Chattanooga', 'Fort Lauderdale', 'Rancho Cucamonga', 'Santa Rosa', 'Port St. Lucie', 'Ontario', 'Tempe', 'Vancouver', 'Springfield', 'Cape Coral', 'Pembroke Pines', 'Sioux Falls', 'Peoria', 'Lancaster', 'Elk Grove', 'Corona', 'Eugene', 'Salem', 'Palmdale', 'Salinas', 'Springfield', 'Pasadena', 'Rockford', 'Pomona', 'Hayward', 'Fort Collins', 'Joliet', 'Escondido', 'Kansas City', 'Torrance', 'Bridgeport', 'Alexandria', 'Sunnyvale', 'Cary', 'Lakewood', 'Hollywood', 'Paterson', 'Syracuse', 'Naperville', 'McKinney', 'Mesquite', 'Clarksville', 'Savannah', 'Dayton', 'Orange', 'Fullerton', 'Pasadena', 'Hampton', 'McAllen', 'Killeen', 'Warren', 'West Valley City', 'Columbia', 'New Haven', 'Sterling Heights', 'Olathe', 'Miramar', 'Thousand Oaks', 'Frisco', 'Cedar Rapids', 'Topeka', 'Visalia', 'Waco', 'Elizabeth', 'Bellevue', 'Gainesville', 'Simi Valley', 'Charleston', 'Carrollton', 'Coral Springs', 'Stamford', 'Hartford', 'Concord', 'Roseville', 'Thornton', 'Kent', 'Lafayette', 'Surprise', 'Denton', 'Victorville', 'Evansville', 'Midland', 'Santa Clara', 'Athens', 'Allentown', 'Abilene', 'Beaumont', 'Vallejo', 'Independence', 'Springfield', 'Ann Arbor', 'Provo', 'Peoria', 'Norman', 'Berkeley', 'El Monte', 'Murfreesboro', 'Lansing', 'Columbia', 'Downey', 'Costa Mesa', 'Inglewood', 'Miami Gardens', '', 'Elgin', 'Wilmington', 'Waterbury', 'Fargo', 'Arvada', 'Carlsbad', 'Westminster', 'Rochester', 'Gresham', 'Clearwater', 'Lowell', 'West Jordan', 'Pueblo', 'San Buenaventura', 'Fairfield', 'West Covina', 'Billings', 'Murrieta', 'High Point', 'Round Rock', 'Richmond', 'Cambridge', 'Norwalk', 'Odessa', 'Antioch', 'Temecula', 'Green Bay', 'Everett', 'Wichita Falls', 'Burbank', 'Palm Bay', 'Centennial', 'Daly City', 'Richardson', 'Pompano Beach', 'Broken Arrow', 'North Charleston', 'West Palm Beach', 'Boulder', 'Rialto', 'Santa Maria', 'El Cajon', 'Davenport', 'Erie', 'Las Cruces', 'South Bend', 'Flint', 'Kenosha']
    #w_emb_2d = pca.fit_transform(w_emb)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree', n_jobs=10).fit(w_emb)
    distances, indices = nbrs.kneighbors(w_emb)
    w_neighbours = defaultdict(list)
    for i in range(indices.shape[0]):
        w = ws[i]
        for j in range(indices.shape[1]):
            w_neighbours[w].append(ws[indices[i, j]])
    
    with open('emnlp2016/w_nbrs.pkl', 'wb') as outf:
        pickle.dump(w_neighbours, outf)
    
    pdb.set_trace()
def tsne_cities():
    from sklearn.decomposition import PCA
    import matplotlib.pylab as plt
    from sklearn import manifold
    pca = PCA(n_components=2)
    
    with open('emnlp2016/city_embeddings.pkl', 'rb') as inf:
        w_emb, ws = pickle.load(inf)
    index_w = dict(zip(range(len(ws)), ws))
    w_index = dict(zip(ws, range(len(ws))))
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, method='exact')
    w_emb_tsne = tsne.fit_transform(w_emb)
    #plt.scatter(w_emb_tsne[:,1], w_emb_tsne[:,0], 20)
    for i, txt in enumerate(ws):
        plt.annotate(txt, (w_emb_tsne[i, 0],w_emb_tsne[i, 1]))
    plt.ylim((-250, +250))
    plt.xlim((-300, +300))
    plt.show()
    #cities_50 = ['nyc','la','Chicago','Houston','Philadelphia','Phoenix','San Antonio','San Diego','Dallas','San Jose','Austin','Jacksonville','San Francisco','Indianapolis','Columbus','Fort Worth','Charlotte','Seattle','Denver','El Paso','Detroit','Washington','Boston','Memphis','Nashville','Portland','Oklahoma','Las Vegas','Baltimore','Louisville','Milwaukee','Albuquerque','Tucson','Fresno','Sacramento','Kansas City','Long Beach','Mesa','Atlanta','Colorado Springs','Virginia Beach','Raleigh','Omaha','Miami','Oakland','Minneapolis','Tulsa','Wichita','New Orleans','Arlington']
    #states_50 = ['California','Texas','Florida','New York','Illinois','Pennsylvania','Ohio','Georgia','North Carolina','Michigan','New Jersey','Virginia','Washington','Arizona','Massachusetts','Indiana','Tennessee','Missouri','Maryland','Wisconsin','Minnesota','Colorado','South Carolina','Alabama','Louisiana','Kentucky','Oregon','Oklahoma','Connecticut','Iowa','Utah','Mississippi','Arkansas','Kansas','Nevada','New Mexico','Nebraska','West Virginia','Idaho','Hawaii','New Hampshire','Maine','Rhode Island','Montana','Delaware','South Dakota','North Dakota','Alaska','Vermont','Wyoming']
    #w_emb_2d = pca.fit_transform(w_emb)
def tsne_states():
    
    from sklearn.decomposition import PCA
    import matplotlib.pylab as plt
    #plt.ion()
    from sklearn import manifold
    pca = PCA(n_components=2)
    f, ax = plt.subplots()
    with open('emnlp2016/state_embeddings.pkl', 'rb') as inf:
        w_emb, ws = pickle.load(inf)
    index_w = dict(zip(range(len(ws)), ws))
    w_index = dict(zip(ws, range(len(ws))))
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, method='exact')
    w_emb_tsne = tsne.fit_transform(w_emb)
    #plt.scatter(w_emb_tsne[:,0], w_emb_tsne[:,1], 20)
    for i, txt in enumerate(ws):
        plt.annotate(txt, (-w_emb_tsne[i, 0],-w_emb_tsne[i, 1]))
    plt.ylim((-600, +600))
    plt.xlim((-200, +200))
    plt.show()
def tsne_citystates():
    
    from sklearn.decomposition import PCA
    import matplotlib.pylab as plt
    #plt.ion()
    from sklearn import manifold
    pca = PCA(n_components=2)
    f, ax = plt.subplots()
    with open('emnlp2016/citystate_embeddings.pkl', 'rb') as inf:
        w_emb, ws = pickle.load(inf)
    index_w = dict(zip(range(len(ws)), ws))
    w_index = dict(zip(ws, range(len(ws))))
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, method='exact')
    w_emb_tsne = tsne.fit_transform(w_emb)
    #plt.scatter(w_emb_tsne[:,0], w_emb_tsne[:,1], 20)
    for i, txt in enumerate(ws):
        plt.annotate(txt, (w_emb_tsne[i, 0],w_emb_tsne[i, 1]))
    plt.ylim((-300, +300))
    plt.xlim((-200, +200))
    plt.show()    

def tsne_allcities():
    import matplotlib as mpl
    #mpl.use("pgf")
    pgf_with_rc_fonts = {
        "font.family": "serif",
        "font.serif": [],                   # use latex default serif font
        "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
        "pgf.texsystem": "pdflatex",
    }
    mpl.rcParams.update(pgf_with_rc_fonts)

    from sklearn.decomposition import PCA
    import matplotlib.pylab as plt
    #plt.ion()
    from sklearn import manifold
    pca = PCA(n_components=2, whiten=True)
    f, ax = plt.subplots()
    with open('emnlp2016/allcities_embeddings.pkl', 'rb') as inf:
        w_emb, ws = pickle.load(inf)
        ws[84] = 'Winston-Salem'
    index_w = dict(zip(range(len(ws)), ws))
    w_index = dict(zip(ws, range(len(ws))))
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, method='exact')
    #w_emb_tsne = tsne.fit_transform(w_emb)
    w_emb_pca = pca.fit_transform(w_emb)
    
 
    #pdb.set_trace()
    #plt.scatter(w_emb_tsne[:,0], w_emb_tsne[:,1], 20)
    for i, txt in enumerate(ws):
        fsize =  max((24 - 2 * i / 5), 14)
        if i>100:
            break
        if i%2 == 0:
            rot = 0
        else:
            rot = 90
        rot = 0
        plt.annotate(txt, (-w_emb_pca[i, 0],w_emb_pca[i, 1]), fontsize=fsize, rotation=rot)
    plt.ylim((-2.5, +2.5))
    plt.xlim((-2.5, +2.5))
    ax.set_aspect('equal')
    #plt.tight_layout()
    plt.axis('off')
    plt.show()
    #plt.savefig('allcities-tsne.pgf')   
if __name__ == '__main__':
    #tsne1()
    #tsne_cities()
    #tsne_states()
    #tsne_citystates() 
    tsne_allcities()  