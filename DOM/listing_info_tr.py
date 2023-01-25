# -*- coding: utf-8 -*-


from pydantic import BaseModel

class listing_info_tr(BaseModel):
    price: int
    price_change_count:int
    area:int
    month_published:int
    year_published:int
    type_group_id:int
    city_id : int
    has_lift:bool
    has_parking_space:bool
