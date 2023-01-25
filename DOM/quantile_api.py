# -*- coding: utf-8 -*-

# Core Pkg
import uvicorn
from fastapi import FastAPI, Request
import xgboost as xgb
import nest_asyncio
from listing_info_eur import listing_info_eur
import time
import csv


nest_asyncio.apply()

# Models
xgb_r_lower = xgb.XGBRegressor()
booster_lower = xgb.Booster()
booster_lower.load_model("eur_xgb_lower_model.json")
xgb_r_lower._Booster = booster_lower


xgb_r_upper = xgb.XGBRegressor()
booster_upper = xgb.Booster()
booster_upper.load_model("eur_xgb_upper_model.json")
xgb_r_upper._Booster = booster_upper

# init app
app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    with open('metrics.csv', 'a', newline='') as f:
        csvwriter = csv.writer(f) 
        csvwriter.writerow([str(start_time), str(process_time)])
    return response

# ML Aspect
@app.post('/predict')
async def predict(features: listing_info_eur):
    recieved = features.dict()
    price = recieved['price']
    price_change_count = recieved['price_change_count']
    area = recieved['area']
    month_published = recieved['month_published']
    year_published = recieved['year_published']
    type_group_id = recieved['type_group_id']
    city_id = recieved['city_id']
    has_lift = recieved['has_lift']
    has_parking_space = recieved['has_parking_space']
    IT = recieved['IT']
    PT = recieved['PT']
    SP = recieved['SP']
    prediction_upper = xgb_r_upper.predict([[price, price_change_count, area, 
                                 month_published, year_published,
                                 type_group_id, city_id, 
                                 has_lift, has_parking_space, 
                                 IT, PT, SP]]).tolist()[0]
    
    prediction_lower = xgb_r_lower.predict([[price, price_change_count, area, 
                                 month_published, year_published,
                                 type_group_id, city_id, 
                                 has_lift, has_parking_space, 
                                 IT, PT, SP]]).tolist()[0]
    
    return {f"Upper bound is:, {prediction_upper}, Lower bound is: {prediction_lower}"} 




if __name__ == '__main__': 
    uvicorn.run(app,host="127.0.0.1",port=8000)
