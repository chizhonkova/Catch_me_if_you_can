import os
import pickle
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


PATH_TO_DATA = '../../data'
AUTHOR = 'Natalya_Chizhonkova' 

SEED = 17
N_JOBS = 1
MAX_DF = 0.3
MIN_DF = 1
NUM_TIME_SPLITS = 10    
SITE_NGRAMS = (1, 4)    
MAX_FEATURES = 100000    
BEST_LOGIT_C = 2.4  
 


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_sparse_features(path_to_train, path_to_test, path_to_site_dict,
                           vectorizer_params):
    times = ['time%s' % i for i in range(1, 11)]
    train_df = pd.read_csv(path_to_train,
                       index_col='session_id', parse_dates=times)
    test_df = pd.read_csv(path_to_test,
                      index_col='session_id', parse_dates=times)


    train_df = train_df.sort_values(by='time1')
    
 
    with open(path_to_site_dict, 'rb') as f:
        site_dict = pickle.load(f)
   
    site_dict['Unknown'] = 0
    
    sites = ['site%s' % i for i in range(1, 11)]
    train_sessions = train_df[sites].fillna(0).astype('int').apply(lambda row: 
                                                     ' '.join([site_dict[i] for i in row]), axis=1).tolist()
    test_sessions = test_df[sites].fillna(0).astype('int').apply(lambda row: 
                                                     ' '.join([site_dict[i] for i in row]), axis=1).tolist()

    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(train_sessions)
    X_test = vectorizer.transform(test_sessions)
    y_train = train_df['target'].astype('int').values

    
    train_times, test_times = train_df[times], test_df[times]
    
    return X_train, X_test, y_train, vectorizer, train_times, test_times


def add_features(times, X_sparse, flag):
    hour = times['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1)
    day = ((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1)
    evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)
    day_of_week = times['time1'].apply(lambda t: t.weekday()).values.reshape(-1, 1)
    sess_duration = (times.max(axis=1) - times.min(axis=1)).astype('timedelta64[s]')\
		   .astype('int').values.reshape(-1, 1)
    start_month = times['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5

    scaler = StandardScaler()
    if (flag):
        duration_scaled = scaler.fit_transform(times['sess_duration'].values.reshape(-1, 1))
        start_month_scaled = scaler.fit_transform(times['start_month'].values.reshape(-1, 1))
    else:
        duration_scaled = scaler.transform(times['sess_duration'].values.reshape(-1, 1))
        start_month_scaled = scaler.transform(times['start_month'].values.reshape(-1, 1))
    
        X = hstack([X_sparse, morning, day, evening, duration_scaled, day_of_week, start_month_scaled])
    return X


with timer('Building sparse site features'):
    X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = \
        prepare_sparse_features(
            path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
            path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
            path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),
            vectorizer_params={'ngram_range': SITE_NGRAMS,
                               'max_features': MAX_FEATURES,
                               'max_df' = MAX_DF,
                               'min_df' = MIN_DF})


with timer('Building additional features'):
    X_train_final = add_features(train_times, X_train_sites, 1)
    X_test_final = add_features(test_times, X_test_sites, 0)


with timer('Cross-validation'):
    time_split = TimeSeriesSplit(n_splits=NUM_TIME_SPLITS)
    logit = LogisticRegression(random_state=SEED, solver='liblinear')

    c_values = [BEST_LOGIT_C]

    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                                  scoring='roc_auc', n_jobs=N_JOBS, cv=time_split, verbose=1)
    logit_grid_searcher.fit(X_train_final, y_train)
    print('CV score', logit_grid_searcher.best_score_)


with timer('Test prediction and submission'):
    test_pred = logit_grid_searcher.predict_proba(X_test_final)[:, 1]
    pred_df = pd.DataFrame(test_pred, index=np.arange(1, test_pred.shape[0] + 1),
                       columns=['target'])
    pred_df.to_csv(f'submission_alice_{AUTHOR}.csv', index_label='session_id')
