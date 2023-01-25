# -*- coding: utf-8 -*-

import os
import joblib
from google.cloud import bigquery
from google.cloud.bigquery.client import Client
import psycopg2
import pandas_gbq
import numpy as np
from numpy import sqrt 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
​
​
​
def pred_dom_tr_xgb():
    ##Establishing connection with BigQuery
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\\ML_Excel_Draft\\ml-bq-readonly.json'
    bq_client = Client()
    
    df_all_tr = pandas_gbq.read_gbq("""  SELECT 
                                            listing_id, area, has_lift, has_parking_space, has_swimming_pool, 
                                            contact_type_id, type_group_id, price, price_currency, 
                                            price_change_count, published_at, year_published, quarter_published, 
                                            month_published, city_id, city_name, country_name 
                                        FROM  ***.***; 
                                    """)
    ​
    ##############################TYPE_CONVERSION##############################

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

    df_all_tr['y_m'] = df_all_tr['published_at'].dt.to_period('M')​
    
    ###############################ONE_HOT_ENCODING################################

    encoder = OneHotEncoder(handle_unknown='ignore')

    encoded_y_m = pd.DataFrame(encoder.fit_transform(df_all_tr[['y_m']]).toarray())

    encoded_y_m.columns = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', 
                           '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
                           '2021-11', '2021-12', '2022-01', '2022-02']

    df_all_tr = df_all_tr.join(encoded_y_m)
    df_all_tr = df_all_tr.drop(columns= ['y_m'], axis= 'columns')

    ##################OUTLIER DETECTION_TURKEY#########################
​
    df_all_tr = df_all_tr [df_all_tr .area < df_all_tr .area.quantile(.95)]
    df_all_tr = df_all_tr[df_all_tr.area > df_all_tr .area.quantile(0.05)]
​
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
    rfm_tr['M_rank_norm'] = (rfm_tr['M_rank']/rfm_tr['M_rank'].max())*100

    rfm_tr.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    df_all_tr = df_all_tr.merge( rfm_tr, how='left', left_on=["city_name"], right_on=["city_name"])
    
    ######################################################################
​
    df_y_tr = df_all_tr[['price', 'price_change_count', 'area', 'type_group_id',
                         'has_lift', 'has_parking_space', 'has_swimming_pool',
                         'R_rank_norm', 'F_rank_norm', 'M_rank_norm', 
                         '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', 
                         '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
                         '2021-11', '2021-12', '2022-01', '2022-02']].copy()
​
    ##Creating Model
    loaded_model = joblib.load('dom_tr_xgb_model.sav')
    yhat_xgb_tr = loaded_model.predict(df_y_tr)
​
    df_y_tr['listing_id'] = df_all_tr['listing_id']
    df_y_tr['dom_predictions'] = yhat_xgb_tr
​
​
    ##Insert predicted listing DOM into analytics.listing_dom_predictions
    conn = psycopg2.connect(host=os.environ['DB_HOST'], port=os.environ['DB_PORT'], 
                            database=os.environ['DB_DATABASE'], user=os.environ['DB_USER'], password=os.environ['DB_PASSWORD'])
    cur = conn.cursor()
    for i, row in df_y_tr.iterrows():
        insert_into_query = ("""INSERT INTO analytics.listing_dom_predictions (listing_id, dom_predictions, created_at)
                                VALUES(%s, %s, current_timestamp);""")    
        row_to_insert = (row['listing_id'], row['dom_predictions'])
        cur.execute(insert_into_query, row_to_insert)
        conn.commit()
    conn.close()
​
​
#pred_dom_tr_xgb()
print("Prod Prediction is done!")
