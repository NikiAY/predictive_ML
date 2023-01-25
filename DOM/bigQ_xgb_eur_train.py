# -*- coding: utf-8 -*-


import os
import time
import pandas_gbq
import numpy as np
from numpy import sqrt 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from xgboost import plot_importance
import matplotlib.pyplot as plt
#from yellowbrick.regressor import ResidualsPlot
#from yellowbrick.regressor import PredictionError
#from yellowbrick.regressor.prediction_error import prediction_error
#from yellowbrick.model_selection import FeatureImportances
#from yellowbrick.model_selection import LearningCurve
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from fast_ml.model_development import train_valid_test_split

import scipy.stats as stats
#import shap


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\\ML_Excel_Draft\\ml-bq-readonly.json'

df_all_eur = pandas_gbq.read_gbq(""" SELECT 
                                        listing_id, area, has_lift, has_parking_space, has_swimming_pool, contact_type_id,
                                        type_group_id, price, price_currency, price_change_count, dom ,published_at, year_published,
                                        quarter_published, month_published, city_id, city_name, country_name 
                                     FROM  ***.*** """)
                                     
#df_sample_eur.to_csv(r'D:\Subpopulation_Analysis\df_sample_eur_z.csv', index = False, header=True, encoding= 'UTF-8')

File_Location_all = 'D:\\ML_Excel_Draft\\df_all_eur.csv'
df_all_eur = pd.read_csv(File_Location_all)

File_Location = 'D:\\ML_Excel_Draft\\preds_25_05.csv'
df_new_eur = pd.read_csv(File_Location)

df_ref_eur = df_all_eur

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

###############################RFM_ANALYSIS_CITIES##########################

frequency_eur_ct = df_all_eur.drop_duplicates().groupby(by=['city_name'], as_index=False)['published_at'].count()
frequency_eur_ct.columns = ['city_name', 'city_freq']

monetary_eur_ct = df_all_eur.groupby(by='city_name', as_index=False)['price'].sum()
monetary_eur_ct.columns = ['city_name', 'city_monetary']


recency_eur_ct = df_all_eur.groupby(by='city_name', as_index=False)['published_at'].max()
recency_eur_ct.columns = ['city_name', 'Last_published_at']

recent_date_ct = recency_eur_ct['Last_published_at'].max()
recency_eur_ct['city_recency'] = recency_eur_ct['Last_published_at'].apply(lambda x: (recent_date_ct - x).days)


rf_eur_ct = recency_eur_ct.merge(frequency_eur_ct, on='city_name')
rfm_eur_ct = rf_eur_ct.merge(monetary_eur_ct, on='city_name').drop(columns='Last_published_at')

rfm_eur_ct['R_rank'] = rfm_eur_ct['city_recency'].rank(ascending=False)
rfm_eur_ct['F_rank'] = rfm_eur_ct['city_freq'].rank(ascending=True)
rfm_eur_ct['M_rank'] = rfm_eur_ct['city_monetary'].rank(ascending=True)

# normalizing the rank of the cities
rfm_eur_ct['R_rank_norm_cities'] = (rfm_eur_ct['R_rank']/rfm_eur_ct['R_rank'].max())*100
rfm_eur_ct['F_rank_norm_cities'] = (rfm_eur_ct['F_rank']/rfm_eur_ct['F_rank'].max())*100
rfm_eur_ct['M_rank_norm_cities'] = (rfm_eur_ct['M_rank']/rfm_eur_ct['M_rank'].max())*100

rfm_eur_ct.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

df_all_eur = df_all_eur.merge( rfm_eur_ct, how='left', left_on=["city_name"], right_on=["city_name"])


###############################RFM_ANALYSIS_TYPE_GROUPS########################

frequency_eur_tg = df_all_eur.drop_duplicates().groupby(by=['type_group_id'], as_index=False)['published_at'].count()
frequency_eur_tg.columns = ['type_group_id', 'type_group_freq']

monetary_eur_tg = df_all_eur.groupby(by='type_group_id', as_index=False)['price'].sum()
monetary_eur_tg.columns = ['type_group_id', 'type_group_monetary']


recency_eur_tg = df_all_eur.groupby(by='type_group_id', as_index=False)['published_at'].max()
recency_eur_tg.columns = ['type_group_id', 'Last_published_at']

recent_date_tg = recency_eur_tg['Last_published_at'].max()
recency_eur_tg['type_group_recency'] = recency_eur_tg['Last_published_at'].apply(lambda x: (recent_date_tg - x).days)


rf_eur_tg = recency_eur_tg.merge(frequency_eur_tg, on='type_group_id')
rfm_eur_tg = rf_eur_tg.merge(monetary_eur_tg, on='type_group_id').drop(columns='Last_published_at')

rfm_eur_tg['R_rank'] = rfm_eur_tg['type_group_recency'].rank(ascending=False)
rfm_eur_tg['F_rank'] = rfm_eur_tg['type_group_freq'].rank(ascending=True)
rfm_eur_tg['M_rank'] = rfm_eur_tg['type_group_monetary'].rank(ascending=True)

# normalizing the rank of the cities
rfm_eur_tg['R_rank_norm_type_group'] = (rfm_eur_tg['R_rank']/rfm_eur_tg['R_rank'].max())*100
rfm_eur_tg['F_rank_norm_type_group'] = (rfm_eur_tg['F_rank']/rfm_eur_tg['F_rank'].max())*100
rfm_eur_tg['M_rank_norm_type_group'] = (rfm_eur_tg['M_rank']/rfm_eur_tg['M_rank'].max())*100

rfm_eur_tg.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

df_all_eur = df_all_eur.merge( rfm_eur_tg, how='left', left_on=["type_group_id"], right_on=["type_group_id"])

###############################ONE_HOT_ENCODING################################

encoder = OneHotEncoder(handle_unknown='ignore')

encoded_country = pd.DataFrame(encoder.fit_transform(df_all_eur[['country_name']]).toarray())

#encoder.categories_

encoded_country.columns = ['IT', 'PT', 'SP']

df_all_eur = df_all_eur.join(encoded_country)
#df_all_eur = df_all_eur.drop(columns= ['y_m'], axis= 'columns')


##############################################################################

df_sample_eur = df_all_eur.sample(200000, random_state= 22)

#df_all_eur = pd.concat([df_train_eur, df_all_eur])

df_X = df_sample_eur[['price', 'price_change_count', 'area', 
                   'month_published', 'year_published', 
                   'type_group_id', 'city_id','has_lift', 
                   'has_parking_space', 'IT', 'PT', 'SP', 'dom']].copy()

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df_X, 
                                                                            target = 'dom', 
                                                                            train_size=0.8, 
                                                                            valid_size=0.1, 
                                                                            test_size=0.1)

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
      "Median Absolute Error", round(median_absolute_error(y_train.to_numpy(), y_train_pred),5))

y_valid_pred = xgb_r.predict(X_valid.to_numpy())
print("Valid performance:", "r2: ",round(r2_score(y_valid.to_numpy(), y_valid_pred), 5), 
      "MAE", round(mean_absolute_error(y_valid.to_numpy(), y_valid_pred), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_valid.to_numpy(), y_valid_pred)),5),
      "Median Absolute Error", round(median_absolute_error(y_valid.to_numpy(), y_valid_pred),5))


yhat_xgb = xgb_r.predict(X_test.to_numpy())
print("Test performance:", "r2: ",round(r2_score(y_test.to_numpy(), yhat_xgb), 5), 
      "MAE", round(mean_absolute_error(y_test.to_numpy(), yhat_xgb), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_test.to_numpy(), yhat_xgb)),5),
      "Median Absolute Error", round(median_absolute_error(y_test.to_numpy(), yhat_xgb),5))

adjusted_r2_test = 1 - (1- xgb_r.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)
print("adjusted r2 test:", adjusted_r2_test)

######MODEL_DUMP
xgb_r._Booster.save_model('D:\\ML_Excel_Draft\\dom_pt_es_it_xgb_model_v2_test.json')

# plot feature importance
xgb.plot_importance(xgb_r)


ref_data_sample = df_all_eur[:15000].sample(1000, random_state=0)



df_sample_eur['country_name'].value_counts()

df_sample_eur["country_name"].replace({"SP": "ES"}, inplace=True)

df_sample_eur['price_bins'] = pd.qcut(df_sample_eur['price'], q= [0, .25, .5, .75, 1.])
df_sample_eur['price_bins'] = df_sample_eur['price_bins'].astype(str)

df_sample_eur['dom_bins'] = pd.qcut(df_sample_eur['dom'], q= [0, .25, .5, .75, 1.])
df_sample_eur['dom_bins'] = df_sample_eur['dom_bins'].astype(str)

df_sample_eur['area_bins'] = pd.qcut(df_sample_eur['area'], q= [0, .25, .5, .75, 1.])
df_sample_eur['area_bins'] = df_sample_eur['area_bins'].astype(str)



df_2020_3 = df_sample_eur[(df_sample_eur['year_published'] == 2020) & (df_sample_eur['quarter_published'] == 3)]
df_2020_3['dom'].to_numpy()
df_2020_3['dom_z'] = stats.zscore(df_2020_3['dom'].astype(float))

df_2020_4 = df_sample_eur[(df_sample_eur['year_published'] == 2020) & (df_sample_eur['quarter_published'] == 4)]
df_2020_4['dom'].to_numpy()
df_2020_4['dom_z'] = stats.zscore(df_2020_4['dom'].astype(float))

df_2020 = pd.concat([df_2020_3, df_2020_4])

df_2021_1 = df_sample_eur[(df_sample_eur['year_published'] == 2021) & (df_sample_eur['quarter_published'] == 1)]
df_2021_1['dom'].to_numpy()
df_2021_1['dom_z'] = stats.zscore(df_2021_1['dom'].astype(float))

df_2021_2 = df_sample_eur[(df_sample_eur['year_published'] == 2021) & (df_sample_eur['quarter_published'] == 2)]
df_2021_2['dom'].to_numpy()
df_2021_2['dom_z'] = stats.zscore(df_2021_2['dom'].astype(float))

df_2021_3 = df_sample_eur[(df_sample_eur['year_published'] == 2021) & (df_sample_eur['quarter_published'] == 3)]
df_2021_3['dom'].to_numpy()
df_2021_3['dom_z'] = stats.zscore(df_2021_3['dom'].astype(float))

df_2021_4 = df_sample_eur[(df_sample_eur['year_published'] == 2021) & (df_sample_eur['quarter_published'] == 4)]
df_2021_4['dom'].to_numpy()
df_2021_4['dom_z'] = stats.zscore(df_2021_4['dom'].astype(float))

df_2021 = pd.concat([df_2021_1, df_2021_2, df_2021_3, df_2021_4])

df_2022_1 = df_sample_eur[(df_sample_eur['year_published'] == 2022) & (df_sample_eur['quarter_published'] == 1)]
df_2022_1['dom'].to_numpy()
df_2022_1['dom_z'] = stats.zscore(df_2022_1['dom'].astype(float))

df_2022_2 = df_sample_eur[(df_sample_eur['year_published'] == 2022) & (df_sample_eur['quarter_published'] == 2)]
df_2022_2['dom'].to_numpy()
df_2022_2['dom_z'] = stats.zscore(df_2022_2['dom'].astype(float))

df_2022 = pd.concat([df_2022_1, df_2022_2])

df_sample_eur = pd.concat([df_2020, df_2021, df_2022])

############################################

df_all_eur.loc[df_all_eur["year_published"] == 2020, "month_published"].value_counts()

df_all_eur.loc[df_all_eur["year_published"] == 2021, "month_published"].value_counts()

df_all_eur.loc[df_all_eur["year_published"] == 2022, "month_published"].value_counts()

df_all_eur.loc[df_all_eur["year_published"] == 2020].max()


############################################

df_sample_eur.loc[df_sample_eur["year_published"] == 2020, "month_published"].value_counts()

df_sample_eur.loc[df_sample_eur["year_published"] == 2021, "month_published"].value_counts()

df_sample_eur.loc[df_sample_eur["year_published"] == 2022, "month_published"].value_counts()

cities_weekday = df_all_eur.groupby(
    [df_all_eur["published_at"].dt.weekday, "city_name"])["dom"].median()


#############################XGB_BAHA#######################################

xgb_b = xgb.XGBRegressor(max_depth=12,
                        subsample=0.33,
                        objective='reg:squarederror',
                        n_estimators=5000,
                        learning_rate = 0.01)
eval_set = [(X_train, y_train), (X_valid, y_valid)]

xgb_b.fit(X_train, y_train, eval_set=eval_set, eval_metric='mae', early_stopping_rounds= 25)

y_train_pred_b = xgb_b.predict(X_train)
print("Train Performance: ", round(r2_score(y_train, y_train_pred_b), 5),
 round(mean_absolute_error(y_train, y_train_pred_b), 5), 
 round(np.sqrt(mean_squared_error(y_train, y_train_pred_b)),5))

yhat_xgb_b = xgb_b.predict(X_test)
print("Test performance:", round(r2_score(y_test, yhat_xgb_b), 5), 
      round(mean_absolute_error(y_test, yhat_xgb_b), 5), 
      round(np.sqrt(mean_squared_error(y_test, yhat_xgb_b)),5))

adjusted_r2_b = 1 - (1- xgb_b.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)
print("adjusted_r2:", adjusted_r2_b)

#############################STACKED_EUR#######################################

level0 = list()
level0.append(('lr', LinearRegression()))
level0.append(('rfr', RandomForestRegressor()))
level0.append(('xgb', xgb.XGBRegressor(max_depth= 12, subsample= 0.33, objective='reg:squarederror', 
                                       n_estimators= 800, learning_rate = 0.01)))
level1 = LinearRegression()

stack_reg = StackingRegressor(estimators=level0, final_estimator=level1, passthrough= True, n_jobs= -1)
stack_reg.fit(X_train, y_train)

y_train_stack_pred = stack_reg.predict(X_train)
print("Train Performance: ", round(r2_score(y_train, y_train_stack_pred), 5),
 round(mean_absolute_error(y_train, y_train_stack_pred), 5), 
 round(np.sqrt(mean_squared_error(y_train, y_train_stack_pred)),5))

yhat_stack = stack_reg.predict(X_test)
print("Test performance:", round(r2_score(y_test, yhat_stack), 5), 
      round(mean_absolute_error(y_test, yhat_stack), 5), 
      round(np.sqrt(mean_squared_error(y_test, yhat_stack)),5))

adjusted_r2_stack = 1 - (1- stack_reg.score(X, y))*(len(y)-1)/(len(y)- X.shape[1]-1)
print("adjusted_r2:", adjusted_r2_stack)


df_check = df_sample_eur
yhat_xgb = pd.Series(yhat_xgb)

df_check = pd.merge(df_sample_eur, yhat_xgb.rename('dom_preds'), how = 'left', left_index = True, right_index = True)

dom = df_check.pop('dom')
df_check.insert(24, 'dom', dom)
#######################################PLOTS

viz_res = ResidualsPlot(xgb_r,
                    train_color="dodgerblue",
                    test_color="tomato",
                    fig=plt.figure(figsize=(9,9))
                    )

viz_res.fit(X_train, y_train)
viz_res.score(X_test, y_test)
viz_res.show()


# =============================================================================
# viz_pred = PredictionError(xgb_r,
#                       fig=plt.figure(figsize=(9,9))
#                       )
# 
# viz_pred.fit(X_train, y_train)
# viz_pred.score(X_test, y_test)
# viz_pred.show()
# =============================================================================

prediction_error(xgb_r,
                 X_train, y_train,
                 X_test, y_test,
                 fig=plt.figure(figsize=(7,7))
                 );

viz_fi = FeatureImportances(xgb_r, relative=False)
viz_fi.fit(X_train, y_train)
viz_fi.show()

viz_lc = LearningCurve(xgb_r, scoring='neg_median_absolute_error', n_jobs= -1)
viz_lc.fit(X, y)       
viz_lc.show()  

plt.hist(df_all_eur['dom'], bins= 60)
plt.xticks(np.arange(0, 600, 10))
plt.show()

explainer = shap.Explainer(xgb_r)
shap_values = explainer(X_test)

shap.plots.bar(shap_values, max_display=25)

explainer_tree = shap.TreeExplainer(xgb_r)
shap_values_tree = explainer_tree.shap_values(X_test)

##########
shap_display = shap.force_plot(explainer_tree.expected_value, shap_values_tree[130], 
                               X_test.iloc[4].values , 
                               feature_names = X_test.columns, matplotlib=True)
display(shap_display)


##########
shap_values_train = shap.TreeExplainer(xgb_r).shap_values(X_train)
shap.summary_plot(shap_values_train, X_train)

shap_values_test = shap.TreeExplainer(xgb_r).shap_values(X_test)
shap.summary_plot(shap_values_test, X_test)


### MODEL_DUMP ###############################################################

filename = 'D:\\ML_Excel_Draft\\dom_pt_es_it_xgb_model_v2.sav'
joblib.dump(xgb_r, filename)
​
print("Training has been done")

# =============================================================================
# y_train_rfr_pred = rfr.predict(X_train)
# print("Train Performance: ", round(r2_score(y_train, y_train_rfr_pred), 5),
#  round(mean_absolute_error(y_train, y_train_rfr_pred), 5), 
#  round(np.sqrt(mean_squared_error(y_train, y_train_rfr_pred)),5))
# 
# yhat_rfr = rfr.predict(X_test)
# print("Test performance:", round(r2_score(y_test, yhat_rfr), 5), 
#       round(mean_absolute_error(y_test, yhat_rfr), 5), 
#       round(np.sqrt(mean_squared_error(y_test, yhat_rfr)),5))
# 
# adjusted_r2_stack = 1 - (1- stack_reg.score(X, y))*(len(y)-1)/(len(y)- X.shape[1]-1)
# print("adjusted_r2:", adjusted_r2_stack)
# 
# =============================================================================

###############DMATRIX

d_train = xgb.DMatrix(data=X_train.values, feature_names= X_train.columns, label= y_train.values)
d_val = xgb.DMatrix(data=X_valid.values, feature_names= X_valid.columns, label= y_valid.values)
d_test = xgb.DMatrix(data=X_test.values, feature_names= X_test.columns, label= y_test.values)

#eval_set = [(X_train, y_train), (X_valid, y_valid)]

#############################XGB_EUROPE#######################################

params = { 
    'objective' :'reg:squarederror', 'reg_lambda': 2.7921 , 
    'reg_alpha': 50, 'eval_metric': 'rmse', 'gamma': 13 , 
    'max_depth': 16, 'min_child_weight': 14.1631, 
    'eta': 0.0604, 'colsample_bytree': 0.9793 , 
    'colsample_bylevel': 0.8327, 'subsample': 0.9029, 
    'verbose_eval': 1, 'seed': 123, 'n_jobs': 16
    }

evals_result = {}

eval_set = [(d_train, 'train'), (d_val, 'eval')] 

num_rounds = 1000

xgb_r = xgb.train(
    params,
    d_train,
    num_boost_round=num_rounds,
    evals = eval_set,
    evals_result=evals_result,
    verbose_eval = True
)

#print(evals_result)


y_train_pred = xgb_r.predict(d_train)
print("Train Performance: ", 
      "r2:",round(r2_score(y_train, y_train_pred), 5), 
      "MAE", round(mean_absolute_error(y_train, y_train_pred), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_train, y_train_pred)),5),
      "Median Absolute Error", round(median_absolute_error(y_train, y_train_pred),5))

y_valid_pred = xgb_r.predict(d_val)
print("Valid performance:", "r2: ",round(r2_score(y_valid, y_valid_pred), 5), 
      "MAE", round(mean_absolute_error(y_valid, y_valid_pred), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_valid, y_valid_pred)),5),
      "Median Absolute Error", round(median_absolute_error(y_valid, y_valid_pred),5))


yhat_xgb = xgb_r.predict(d_test)
print("Test performance:", "r2: ",round(r2_score(y_test, yhat_xgb), 5), 
      "MAE", round(mean_absolute_error(y_test, yhat_xgb), 5), 
      "MSE", round(np.sqrt(mean_squared_error(y_test, yhat_xgb)),5),
      "Median Absolute Error", round(median_absolute_error(y_test, yhat_xgb),5))

adjusted_r2_test = 1 - (1- r2_score(y_test, yhat_xgb))*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)
print("adjusted r2 test:", adjusted_r2_test)
