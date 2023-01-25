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
#import hyperopt
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

##Establishing connection with BigQuery Fizbot DWH and getting train data
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '.env\\ml-bq-readonly.json'

df_all_eur = pandas_gbq.read_gbq(""" SELECT 
                                        listing_id, area, has_lift, has_parking_space, has_swimming_pool, contact_type_id,
                                        type_group_id, price, price_currency, price_change_count, dom ,published_at, year_published,
                                        quarter_published, month_published, city_id, city_name, country_name 
                                     FROM ***.*** """)
                                     
#File_Location_all = 'D:\\ML_Excel_Draft\\df_all_eur.csv'
#df_all_eur = pd.read_csv(File_Location_all)

#df_ref_eur = df_all_eur

#df_all_eur.to_csv('df_all_eur.csv', index=False, encoding= 'utf-8') 


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

df_sample_eur = df_all_eur.sample(400000, random_state= 34)


df_X = df_all_eur[['price', 'price_change_count', 'area', 
                   'month_published', 'year_published', 
                   'type_group_id', 'city_id','has_lift', 
                   'has_parking_space', 'IT', 'PT', 'SP',
                   'dom']].copy()

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df_X, 
                                                                            target = 'dom', 
                                                                            train_size=0.7, 
                                                                            valid_size=0.15, 
                                                                            test_size=0.15)

#######LOG_COSH_QUANTILE

def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

alpha = 0.95


#############################################################

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
                           colsample_bylevel= space['colsample_bylevel'], 
                           objective= log_cosh_quantile(1 - alpha), n_jobs= 16)

    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    model.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="mae",
            early_stopping_rounds=10, verbose=True)

    pred = model.predict(X_test)
    median_ae = median_absolute_error(y_test, pred)
    print ("SCORE:", median_ae)

    return {'loss': median_ae, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials= trials)

print (best)


#############################XGB_EUROPE_UPPER_BOUND#######################################

xgb_r_upper = xgb.XGBRegressor(objective = log_cosh_quantile(alpha), n_estimators = 500, reg_lambda= 2.4755, 
                         reg_alpha= 45.273, gamma= 3.4073 , max_depth= 22, min_child_weight= 68.7668, 
                         eta= 0.03754, colsample_bytree= 0.9250 , colsample_bylevel= 0.8932, 
                         subsample=  0.6204, seed = 123, n_jobs= 16)

eval_set = [(X_train.to_numpy(), y_train.to_numpy()), (X_valid.to_numpy(), y_valid.to_numpy())]

train_start = time.time()

xgb_r_upper.fit(X_train.to_numpy(), y_train.to_numpy(), 
          eval_set=eval_set, eval_metric='mae', early_stopping_rounds= 15)

train_stop = time.time()
train_total = train_stop - train_start
print('Training took',round(train_total, 3), 'seconds')


y_train_upper = xgb_r_upper.predict(X_train.to_numpy())
print("Train Performance: ", 
      "r2:",round(r2_score(y_train.to_numpy(), y_train_upper), 5), 
      "MAE", round(mean_absolute_error(y_train.to_numpy(), y_train_upper), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_train.to_numpy(), y_train_upper)),5),
      "Median Absolute Error", round(median_absolute_error(y_train.to_numpy(), y_train_upper),5),
      f"MAPE {mean_absolute_percentage_error(y_train.to_numpy(), y_train_upper)}")

y_valid_upper = xgb_r_upper.predict(X_valid.to_numpy())
print("Valid performance:", "r2: ",round(r2_score(y_valid.to_numpy(), y_valid_upper), 5), 
      "MAE", round(mean_absolute_error(y_valid.to_numpy(), y_valid_upper), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_valid.to_numpy(), y_valid_upper)),5),
      "Median Absolute Error", round(median_absolute_error(y_valid.to_numpy(), y_valid_upper),5),
      f"MAPE {mean_absolute_percentage_error(y_valid.to_numpy(), y_valid_upper)}")


yhat_upper = xgb_r_upper.predict(X_test.to_numpy())
print("Test performance:", "r2: ",round(r2_score(y_test.to_numpy(), yhat_upper), 5), 
      "MAE", round(mean_absolute_error(y_test.to_numpy(), yhat_upper), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_test.to_numpy(), yhat_upper)),5),
      "Median Absolute Error", round(median_absolute_error(y_test.to_numpy(), yhat_upper),5),
      f"MAPE {mean_absolute_percentage_error(y_test.to_numpy(), yhat_upper)}")

adjusted_r2_test = 1 - (1- xgb_r_upper.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)
print("adjusted r2 test:", adjusted_r2_test)


xgb_r_upper._Booster.save_model('eur_xgb_upper_model.json')
xgb_r_upper.name = 'eur_xgb_upper_model'

#############################XGB_EUROPE_LOWER_BOUND#######################################

# =============================================================================
# {'colsample_bylevel': 0.7923172535850423, 'colsample_bytree': 0.9393512850358599, 'eta': 0.15651957891599544, 
#  'gamma': 9.45353724650541, 'max_depth': 5, 'min_child_weight': 59.61027512606985, 'n_estimators': 89, 
#  'reg_alpha': 12.049121037830382, 'reg_lambda': 1.5918712404771371, 'subsample': 0.5025429317920757}
# 
# =============================================================================
xgb_r_lower = xgb.XGBRegressor(objective = log_cosh_quantile(1 - alpha), n_estimators = 200, reg_lambda= 1.5918 , 
                         reg_alpha= 50, gamma= 9.4535 , max_depth= 5, min_child_weight= 59.6102, 
                         eta= 0.1565, colsample_bytree= 0.9393 , colsample_bylevel= 0.7923, 
                         subsample= 0.5025, seed = 123, n_jobs= 16)

eval_set = [(X_train.to_numpy(), y_train.to_numpy()), (X_valid.to_numpy(), y_valid.to_numpy())]

train_start = time.time()

xgb_r_lower.fit(X_train.to_numpy(), y_train.to_numpy(), 
          eval_set=eval_set, eval_metric='mae', early_stopping_rounds= 15)

train_stop = time.time()
train_total = train_stop - train_start
print('Training took',round(train_total, 3), 'seconds')


y_train_lower = xgb_r_lower.predict(X_train.to_numpy())
print("Train Performance: ", 
      "r2:",round(r2_score(y_train.to_numpy(), y_train_upper), 5), 
      "MAE", round(mean_absolute_error(y_train.to_numpy(), y_train_upper), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_train.to_numpy(), y_train_upper)),5),
      "Median Absolute Error", round(median_absolute_error(y_train.to_numpy(), y_train_upper),5),
      f"MAPE {mean_absolute_percentage_error(y_train.to_numpy(), y_train_upper)}")

y_valid_lower = xgb_r_lower.predict(X_valid.to_numpy())
print("Valid performance:", "r2: ",round(r2_score(y_valid.to_numpy(), y_valid_lower), 5), 
      "MAE", round(mean_absolute_error(y_valid.to_numpy(), y_valid_lower), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_valid.to_numpy(), y_valid_lower)),5),
      "Median Absolute Error", round(median_absolute_error(y_valid.to_numpy(), y_valid_lower),5),
      f"MAPE {mean_absolute_percentage_error(y_valid.to_numpy(), y_valid_lower)}")


yhat_lower = xgb_r_lower.predict(X_test.to_numpy())
print("Test performance:", "r2: ",round(r2_score(y_test.to_numpy(), yhat_lower), 5), 
      "MAE", round(mean_absolute_error(y_test.to_numpy(), yhat_lower), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_test.to_numpy(), yhat_lower)),5),
      "Median Absolute Error", round(median_absolute_error(y_test.to_numpy(), yhat_lower),5),
      f"MAPE {mean_absolute_percentage_error(y_test.to_numpy(), yhat_lower)}")

adjusted_r2_test = 1 - (1- xgb_r_lower.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)
print("adjusted r2 test:", adjusted_r2_test)

xgb_r_lower._Booster.save_model('eur_xgb_lower_model.json')
xgb_r_lower.name = 'eur_xgb_lower_model'

##############################################

res = pd.DataFrame({'lower_bound' : yhat_lower, 'true_dom': y_test, 'upper_bound': yhat_upper})
res['interval_length'] = res['upper_bound'] - res['lower_bound']
res['within_interval'] = np.where((res['lower_bound'] <= res['true_dom']) & (res['true_dom'] <= res['upper_bound']), True, False)

conditions = [
    (res['true_dom'] < res['lower_bound']),
    (res['true_dom'] > res['upper_bound']),
    (res['true_dom'] >= res['lower_bound']) & (res['true_dom'] <= res['upper_bound'])]
values = [(abs(res['lower_bound'] - res['true_dom'])), (abs(res['upper_bound'] - res['true_dom'])), 0]
res['diff'] = np.select(conditions, values)

df_preds['diff'].quantile([.05 ,.15 ,.25, .5, .75, .85, .90 ,.95])

df_y = X_test
df_y['listing_id'] = df_all_eur['listing_id']
df_y['country_name'] = df_all_eur['country_name']
df_y['city_name'] = df_all_eur['city_name']
df_y['published_at'] = df_all_eur['published_at']
df_y['true_dom'] = y_test

df_preds = pd.merge(df_y, res[['lower_bound', 'upper_bound', 
                               'interval_length', 'within_interval', 'diff']],
                    how = 'left',left_index = True, right_index = True)
df_y.info()

res.value_counts()

count = res[(res.true_dom >= res.lower_bound) & (res.true_dom <= res.upper_bound)].shape[0]
total = res.shape[0]
print(f'pref = {count / total}')

res['within_interval'].value_counts()
res['diff'].median()

df_preds.to_csv('quantile.csv', index=False, encoding= 'utf-8') 

################################

import matplotlib.pyplot as plt

max_length = 100
fig = plt.figure()
plt.plot(list(y_test[:max_length]), 'gx', label=u'real value')
plt.plot(yhat_upper[:max_length], 'y_', label=u'Q up')
plt.plot(yhat_lower[:max_length], 'b_', label=u'Q low')
index = np.array(range(0, len(yhat_upper[:max_length])))
plt.fill(np.concatenate([index, index[::-1]]),
         np.concatenate([yhat_upper[:max_length], yhat_lower[:max_length][::-1]]),
         alpha=.5, fc='b', ec='None', label='90% prediction interval')
plt.xlabel('$index$')
plt.ylabel('$dom$')
plt.legend(loc='upper left')
plt.show()


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
