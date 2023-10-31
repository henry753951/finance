import datetime as dt
import pandas as pd
class Stock_info:
    stock_id: str
    stock_name: str
    industry_category: str
    type_: str
    up_date: dt.datetime
    
    data : pd.DataFrame
    def __str__(self):
        return f'{self.stock_id} {self.stock_name} {self.date}'

    

    
class stock_daily:
    Stock_info : Stock_info
    date : dt.datetime
    close : float
    
