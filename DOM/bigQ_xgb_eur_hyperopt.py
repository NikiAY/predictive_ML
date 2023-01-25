# -*- coding: utf-8 -*-


import os
from google.cloud import bigquery
from google.cloud.bigquery.client import Client
import pandas_gbq
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import sqrt 
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\\ML_Excel_Draft\\ml-bq-readonly.json'
bq_client = Client()

df_all_eur = pandas_gbq.read_gbq(""" SELECT 
                                        listing_id, area, has_lift, has_parking_space, has_swimming_pool, contact_type_id,
                                        type_group_id, price, price_currency, price_change_count, dom ,published_at, year_published,
                                        quarter_published, month_published, city_id, city_name, country_name 
                                     FROM  ***.*** """)


###############################################################################

df_all_eur['price'] = df_all_eur['price'].astype(int)
df_all_eur['area'] = df_all_eur['area'].astype(int)
df_all_eur['price_change_count'] = df_all_eur['price_change_count'].astype(int)
df_all_eur['city_id'] = df_all_eur['city_id'].astype(int)
df_all_eur['type_group_id'] = df_all_eur['type_group_id'].astype(int)
df_all_eur['contact_type_id'] = df_all_eur['contact_type_id'].astype(int)
df_all_eur['has_lift'] = df_all_eur['has_lift'].astype(bool)
df_all_eur['has_parking_space'] = df_all_eur['has_parking_space'].astype(bool)
df_all_eur['has_swimming_pool'] = df_all_eur['has_swimming_pool'].astype(bool)

df_all_eur['dom'] = df_all_eur['dom'].astype(int)

df_all_eur['published_at'] = pd.to_datetime(df_all_eur['published_at'])
df_all_eur['year_published'] = df_all_eur['year_published'].astype(int)
df_all_eur['quarter_published'] = df_all_eur['quarter_published'].astype(int)
df_all_eur['month_published'] = df_all_eur['month_published'].astype(int)

df_all_eur['y_m'] = df_all_eur['published_at'].dt.to_period('M')

###############################ONE_HOT_ENCODING################################

encoder = OneHotEncoder(handle_unknown='ignore')

encoded_y_m = pd.DataFrame(encoder.fit_transform(df_all_eur[['y_m']]).toarray())

#encoder.categories_

encoded_y_m.columns = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', 
                       '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
                       '2021-11', '2021-12', '2022-01', '2022-02']

df_all_eur = df_all_eur.join(encoded_y_m)
df_all_eur = df_all_eur.drop(columns= ['y_m'], axis= 'columns')


###############################RFM_ANALYSIS####################################

frequency_eur = df_all_eur.drop_duplicates().groupby(
	by=['city_name'], as_index=False)['published_at'].count()
frequency_eur.columns = ['city_name', 'city_freq']

monetary_eur = df_all_eur.groupby(by='city_name', as_index=False)['price'].sum()
monetary_eur.columns = ['city_name', 'city_monetary']


recency_eur = df_all_eur.groupby(by='city_name', as_index=False)['published_at'].max()
recency_eur.columns = ['city_name', 'Last_published_at']

recent_date = recency_eur['Last_published_at'].max()
recency_eur['city_recency'] = recency_eur['Last_published_at'].apply(lambda x: (recent_date - x).days)


rf_eur = recency_eur.merge(frequency_eur, on='city_name')
rfm_eur = rf_eur.merge(monetary_eur, on='city_name').drop(columns='Last_published_at')

rfm_eur['R_rank'] = rfm_eur['city_recency'].rank(ascending=False)
rfm_eur['F_rank'] = rfm_eur['city_freq'].rank(ascending=True)
rfm_eur['M_rank'] = rfm_eur['city_monetary'].rank(ascending=True)

# normalizing the rank of the cities
rfm_eur['R_rank_norm'] = (rfm_eur['R_rank']/rfm_eur['R_rank'].max())*100
rfm_eur['F_rank_norm'] = (rfm_eur['F_rank']/rfm_eur['F_rank'].max())*100
rfm_eur['M_rank_norm'] = (rfm_eur['F_rank']/rfm_eur['M_rank'].max())*100

rfm_eur.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

df_all_eur = df_all_eur.merge( rfm_eur, how='left', left_on=["city_name"], right_on=["city_name"])


#############################################################

X = df_all_eur[['price', 'price_change_count', 'area', 
                'type_group_id', 'contact_type_id',
                'has_lift', 'has_parking_space', 'has_swimming_pool',
                'R_rank_norm', 'F_rank_norm', 'M_rank_norm', 
                '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', 
                '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
                '2021-11', '2021-12', '2022-01', '2022-02']]

y = df_all_eur['dom']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .30, random_state=50)

############################DEFINING_SPACE####################################

space={'max_depth': hp.choice('max_depth', np.arange(3, 30, 1, dtype=int)),
       'eta': hp.uniform('eta', 0.01, 0.2),
        'gamma': hp.uniform ('gamma', 0, 20),
        'subsample': hp.uniform('subsample', 0.1, 1),
        'reg_alpha' : hp.uniform('reg_alpha', 1, 200),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 10),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel' : hp.uniform('colsample_bylevel', 0.1, 1),
        'min_child_weight' : hp.uniform('min_child_weight', 1, 100),
        'n_estimators': hp.choice('n_estimators', np.arange(100, 1000, 1, dtype=int))}


def hyperparameter_tuning(space):
    model=xgb.XGBRegressor(n_estimators =space['n_estimators'],
                           eta =space['eta'], subsample =space['subsample'], 
                           max_depth = space['max_depth'], gamma = space['gamma'], 
                           reg_alpha = space['reg_alpha'], reg_lambda = space['reg_lambda'],
                           min_child_weight= space['min_child_weight'], 
                           colsample_bytree= space['colsample_bytree'], 
                           colsample_bylevel= space['colsample_bylevel'], n_jobs= 16)

    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    model.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="mae",
            early_stopping_rounds=10,verbose=False)

    pred = model.predict(X_test)
    mae= mean_absolute_error(y_test, pred)
    print ("SCORE:", mae)

    return {'loss':mae, 'status': STATUS_OK, 'model': model}

#############################################################

trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials= trials)

print (best)
