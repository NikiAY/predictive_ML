# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import uuid
import hashlib
import pandas_gbq
from google.cloud import bigquery
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from fast_ml.model_development import train_valid_test_split


##Establishing connection with BigQuery 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '.env\\ml-bq-readonly.json'

df_all_eur = pandas_gbq.read_gbq(""" SELECT 
                                        listing_id, area, has_lift, has_parking_space, has_swimming_pool, contact_type_id,
                                        type_group_id, price, price_currency, price_change_count, dom ,published_at, year_published,
                                        quarter_published, month_published, city_id, city_name, country_name 
                                     FROM  ***.*** """)
                                     
#File_Location_all = 'D:\\ML_Excel_Draft\\df_all_eur.csv'
#df_all_eur = pd.read_csv(File_Location_all)

#df_ref_eur = df_all_eur

###############################################################################

df_all_eur[['price', 'area', 'price_change_count', 
            'month_published', 'year_published', 
            'city_id', 'type_group_id', 'contact_type_id', 
            'dom']] = df_all_eur[['price', 'area', 'price_change_count',
                                    'month_published','year_published', 
                                    'city_id', 'type_group_id', 'contact_type_id',
                                    'dom'  ]].astype(int)

df_all_eur[['has_lift', 'has_parking_space', 'has_swimming_pool']] = df_all_eur[['has_lift', 
                                                                                 'has_parking_space', 'has_swimming_pool']].astype(bool)

df_all_eur['published_at'] = pd.to_datetime(df_all_eur['published_at'])


###################OUTLIER DETECTION_EUROPE####################################

df_all_eur = df_all_eur[df_all_eur.area < df_all_eur.area.quantile(.95)]
df_all_eur = df_all_eur[df_all_eur.area > df_all_eur.area.quantile(0.05)]

df_all_eur = df_all_eur[df_all_eur.price < df_all_eur.price.quantile(.95)]
df_all_eur = df_all_eur[df_all_eur.price > df_all_eur.price.quantile(0.05)]


###############################ONE_HOT_ENCODING################################

encoder = OneHotEncoder(handle_unknown='ignore')
encoded_country = pd.DataFrame(encoder.fit_transform(df_all_eur[['country_name']]).toarray())

#encoder.categories_

encoded_country.columns = ['IT', 'PT', 'SP']
df_all_eur = df_all_eur.join(encoded_country)

##############################################################################

df_sample_eur = df_all_eur.sample(100000, random_state= 22)


df_X = df_sample_eur[['price', 'price_change_count', 'area', 
                   'month_published', 'year_published', 
                   'type_group_id', 'city_id','has_lift', 
                   'has_parking_space', 'IT', 'PT', 'SP',
                   'dom']].copy()

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df_X, 
                                                                            target = 'dom', 
                                                                            train_size=0.7, 
                                                                            valid_size=0.15, 
                                                                            test_size=0.15)

#############################XGB_EUROPE#######################################

xgb_r = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 1000, reg_lambda= 2.7921 , 
                         reg_alpha= 50, gamma= 13 , max_depth= 16, min_child_weight= 14.1631, 
                         eta= 0.0604,colsample_bytree= 0.9793 , colsample_bylevel= 0.8327, 
                         subsample= 0.9029, seed = 123, n_jobs= 16)

eval_set = [(X_train.to_numpy(), y_train.to_numpy()), (X_valid.to_numpy(), y_valid.to_numpy())]

train_start = time.time()
xgb_r.fit(X_train.to_numpy(), y_train.to_numpy(), 
          eval_set=eval_set, eval_metric='mae', early_stopping_rounds= 15)
train_stop = time.time()


train_total = train_stop - train_start
print('Training took',round(train_total, 3), 'seconds')

y_train_pred = xgb_r.predict(X_train.to_numpy())
print("Train Performance: ", 
      "r2:",round(r2_score(y_train.to_numpy(), y_train_pred), 5), 
      "MAE", round(mean_absolute_error(y_train.to_numpy(), y_train_pred), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_train.to_numpy(), y_train_pred)),5),
      "Median Absolute Error", round(median_absolute_error(y_train.to_numpy(), y_train_pred),5),
      f"MAPE {mean_absolute_percentage_error(y_train.to_numpy(), y_train_pred)}")

y_valid_pred = xgb_r.predict(X_valid.to_numpy())
print("Valid performance:", "r2: ",round(r2_score(y_valid.to_numpy(), y_valid_pred), 5), 
      "MAE", round(mean_absolute_error(y_valid.to_numpy(), y_valid_pred), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_valid.to_numpy(), y_valid_pred)),5),
      "Median Absolute Error", round(median_absolute_error(y_valid.to_numpy(), y_valid_pred),5),
      f"MAPE {mean_absolute_percentage_error(y_valid.to_numpy(), y_valid_pred)}")


yhat_xgb = xgb_r.predict(X_test.to_numpy())
print("Test performance:", "r2: ",round(r2_score(y_test.to_numpy(), yhat_xgb), 5), 
      "MAE", round(mean_absolute_error(y_test.to_numpy(), yhat_xgb), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_test.to_numpy(), yhat_xgb)),5),
      "Median Absolute Error", round(median_absolute_error(y_test.to_numpy(), yhat_xgb),5),
      f"MAPE {mean_absolute_percentage_error(y_test.to_numpy(), yhat_xgb)}")

adjusted_r2_test = 1 - (1- xgb_r.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)
print("adjusted r2 test:", adjusted_r2_test)


######MODEL_DUMP
xgb_r._Booster.save_model('dom_pt_es_it_xgb_model_v2.json')
xgb_r.name = 'dom_pt_es_it_xgb_model_v2'
#####Saving_MetaData

now = datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')

#id_rand = uuid.uuid4()
#id_rand = id_rand.hex

meta_dict = {}

meta_dict['model_name'] = xgb_r.name
meta_dict['model_type'] = type(xgb_r).__name__
#meta_dict['model_version'] = id_rand
#meta_dict['model_version'] = str(id(xgb_r)
meta_dict['model_version'] = hashlib.sha256(str('dom_pt_es_it_xgb_model_v2').encode('utf-8')).hexdigest()
meta_dict['r2_score'] = round(r2_score(y_test.to_numpy(), yhat_xgb), 5)
meta_dict['adjusted_r2_score'] = adjusted_r2_test
meta_dict['mean_absolute_error'] = round(mean_absolute_error(y_test.to_numpy(), yhat_xgb), 5)
meta_dict['median_absolute_error'] = round(median_absolute_error(y_test.to_numpy(), yhat_xgb),5)
meta_dict['mean_squared_error'] = round(np.sqrt(mean_squared_error(y_test.to_numpy(), yhat_xgb)),5)
meta_dict['train_time'] = train_total
meta_dict['created_at'] = now

#####BigQ_MetaData_Insertion

client = bigquery.Client()

dataset_id = '***'
table_id = 'ml_logs'
table_ref = client.dataset(dataset_id).table(table_id)
table = client.get_table(table_ref)


insertion = client.insert_rows(table, [meta_dict]) 
print(insertion)


print(meta_dict)


import random
import threading
import psutil

CPU = []

def display_cpu():
    global running

    running = True

    currentProcess = psutil.Process()

    # start loop
    while running:
        cpu_usage = currentProcess.cpu_percent(interval=1) / psutil.cpu_count()
        CPU.append(round(cpu_usage, 2))
        print("Current CPU usage is: ", cpu_usage)

def start():
    global t

    # create thread and start it
    t = threading.Thread(target=display_cpu)
    t.start()

def stop():
    global running
    global t

    # use `running` to stop loop in thread so thread will end
    running = False

    # wait for thread's end
    t.join()


# ---

start()
try:
    result = xgb_r.fit(X_train.to_numpy(), y_train.to_numpy(), 
              eval_set=eval_set, eval_metric='mae', early_stopping_rounds= 15)
finally: # stop thread even if I press Ctrl+C
    stop()

print(CPU)
