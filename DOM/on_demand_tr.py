# -*- coding: utf-8 -*-

import uvicorn
from fastapi import FastAPI, Request
import time
import xgboost as xgb
import nest_asyncio
from listing_info_tr import listing_info_tr

nest_asyncio.apply()


# Models
xgb_r = xgb.XGBRegressor()
booster = xgb.Booster()
booster.load_model("dom_tr_xgb_model_v2.json")
xgb_r._Booster = booster


# init app
app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# ML Aspect
@app.post('/predict')
async def predict(features: listing_info_tr):
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

    prediction = xgb_r.predict([[price, price_change_count, area, 
                                 month_published, year_published,
                                 type_group_id, city_id, 
                                 has_lift, has_parking_space]]).tolist()[0] 
    return {"prediction":prediction}



if __name__ == '__main__': 
    uvicorn.run(app,host="127.0.0.2",port=8000)

#hypercorn app:app --bind 127.0.0.2:8000
