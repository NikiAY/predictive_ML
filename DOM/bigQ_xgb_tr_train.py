# -*- coding: utf-8 -*-

import os
import joblib
from google.cloud import bigquery
from google.cloud.bigquery.client import Client
import pandas_gbq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

##Establishing connection with BigQuery
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\\ML_Excel_Draft\\ml-bq-readonly.json'
bq_client = Client()

df_all_tr = pandas_gbq.read_gbq(""" SELECT 
                                        listing_id, area, has_lift, has_parking_space, 
                                        has_swimming_pool, contact_type_id,
                                        type_group_id, price, price_currency, price_change_count, 
                                        dom ,published_at, year_published,
                                        quarter_published, month_published, city_id, city_name, country_name 
                                     FROM  ***.*** """)


#############################################################

df_all_tr['area'] = df_all_tr['area'].astype(int)

df_all_tr['price'] = df_all_tr['price'].astype(int)
df_all_tr['price_change_count'] = df_all_tr['price_change_count'].astype(int)

df_all_tr['city_id'] = df_all_tr['city_id'].astype(int)
df_all_tr['type_group_id'] = df_all_tr['type_group_id'].astype(int)
df_all_tr['contact_type_id'] = df_all_tr['contact_type_id'].astype(int)

df_all_tr['published_at'] = pd.to_datetime(df_all_tr['published_at'])
df_all_tr['year_published'] = df_all_tr['year_published'].astype(int)
df_all_tr['quarter_published'] = df_all_tr['quarter_published'].astype(int)
df_all_tr['month_published'] = df_all_tr['month_published'].astype(int)

df_all_tr['has_lift'] = df_all_tr['has_lift'].astype(bool)
df_all_tr['has_parking_space'] = df_all_tr['has_parking_space'].astype(bool)
df_all_tr['has_swimming_pool'] = df_all_tr['has_swimming_pool'].astype(bool)

df_all_tr['dom'] = df_all_tr['dom'].astype(int)

df_all_tr['y_m'] = df_all_tr['published_at'].dt.to_period('M')

###############################ONE_HOT_ENCODING################################

encoder = OneHotEncoder(handle_unknown='ignore')

encoded_y_m = pd.DataFrame(encoder.fit_transform(df_all_tr[['y_m']]).toarray())

encoded_y_m.columns = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', 
                       '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
                       '2021-11', '2021-12', '2022-01', '2022-02']

df_all_tr = df_all_tr.join(encoded_y_m)
df_all_tr = df_all_tr.drop(columns= ['y_m'], axis= 'columns')

###################OUTLIER DETECTION_TURKEY####################################

df_all_tr = df_all_tr [df_all_tr .area < df_all_tr .area.quantile(.95)]
df_all_tr = df_all_tr[df_all_tr.area > df_all_tr .area.quantile(0.05)]


df_all_tr = df_all_tr[df_all_tr.price < df_all_tr.price.quantile(.95)]
df_all_tr = df_all_tr[df_all_tr.price > df_all_tr.price.quantile(0.05)]

###############################RFM_ANALYSIS####################################

frequency_tr = df_all_tr.drop_duplicates().groupby( by=['city_name'], as_index=False)['published_at'].count()
frequency_tr.columns = ['city_name', 'city_freq']

monetary_tr = df_all_tr.groupby(by='city_name', as_index=False)['price'].sum()
monetary_tr.columns = ['city_name', 'city_monetary']


recency_tr = df_all_tr.groupby(by='city_name', as_index=False)['published_at'].max()
recency_tr.columns = ['city_name', 'Last_published_at']

recent_date = recency_tr['Last_published_at'].max()
recency_tr['city_recency'] = recency_tr['Last_published_at'].apply(lambda x: (recent_date - x).days)


rf_tr = recency_tr.merge(frequency_tr, on='city_name')
rfm_tr = rf_tr.merge(monetary_tr, on='city_name').drop(columns='Last_published_at')

rfm_tr['R_rank'] = rfm_tr['city_recency'].rank(ascending=False)
rfm_tr['F_rank'] = rfm_tr['city_freq'].rank(ascending=True)
rfm_tr['M_rank'] = rfm_tr['city_monetary'].rank(ascending=True)

# normalizing the rank of the cities
rfm_tr['R_rank_norm'] = (rfm_tr['R_rank']/rfm_tr['R_rank'].max())*100
rfm_tr['F_rank_norm'] = (rfm_tr['F_rank']/rfm_tr['F_rank'].max())*100
rfm_tr['M_rank_norm'] = (rfm_tr['F_rank']/rfm_tr['M_rank'].max())*100

rfm_tr.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

df_all_tr = df_all_tr.merge( rfm_tr, how='left', left_on=["city_name"], right_on=["city_name"])


#############################################################

X = df_all_tr[['price', 'price_change_count', 'area', 'type_group_id',
                'has_lift', 'has_parking_space', 'has_swimming_pool',
                   'R_rank_norm', 'F_rank_norm', 'M_rank_norm', 
                   '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', 
                   '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
                   '2021-11', '2021-12', '2022-01', '2022-02']]
y = df_all_tr['dom']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .30, random_state=50)

#############################################################

#####XGB_TURKEY

xgb_r = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 600, reg_lambda= 7.8634 , 
                         reg_alpha= 9.6275, gamma= 7.9183, max_depth= 9 , min_child_weight= 17.57 , 
                         eta= 0.1886, colsample_bytree= 0.6304 , colsample_bylevel= 0.2338, subsample= 0.93952, 
                         seed = 123, n_jobs= 16)
xgb_r.fit(X, y)

y_train_pred = xgb_r.predict(X_train)
print("Train Performance: ", round(r2_score(y_train, y_train_pred), 5),
 round(mean_absolute_error(y_train, y_train_pred), 5), 
 round(np.sqrt(mean_squared_error(y_train, y_train_pred)),5))

yhat_xgb = xgb_r.predict(X_test)
print("Test performance:", round(r2_score(y_test, yhat_xgb), 5), 
      round(mean_absolute_error(y_test, yhat_xgb), 5), 
      round(np.sqrt(mean_squared_error(y_test, yhat_xgb)),5))

adjusted_r2 = 1 - (1- xgb_r.score(X, y))*(len(y)-1)/(len(y)- X.shape[1]-1)
print("Adjusted_r2:", adjusted_r2)


### MODEL_DUMP ###############################################################

filename = 'dom_tr_xgb_model.sav'
joblib.dump(xgb_r, filename)
​
print("Training has been done")
