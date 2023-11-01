import pandas as pd
import datetime as dt
from model import Stock_info
from FinMind.data import DataLoader
import time

# import dotenv
# dotenv.load_dotenv()


def get_all_stock(start_id: int, end_id: int) -> list[Stock_info]:
    df = pd.read_csv('data/taiwan_stock_info.csv')
    stocks = []
    for _, row in df.iterrows():
        try:
            if int(row['stock_id']) < start_id or int(row['stock_id']) > end_id:
                continue
            if row['stock_id'] != str(int(row['stock_id'])):
                continue
        except ValueError:
            continue
        stock = Stock_info()
        stock.stock_id = str(int(row['stock_id']))
        stock.stock_name = row['stock_name']
        stock.industry_category = row['industry_category']
        stock.type_ = row['type']
        stock.up_date = pd.to_datetime(row['date'])
        stocks.append(stock)
    return stocks


def get_stock_daily(stock_id: int) -> pd.DataFrame:
    api = DataLoader()
    # api.login(user_id="",password='password')
    df = api.taiwan_stock_daily(
        stock_id=F'{stock_id}',
        start_date='2000-01-01',
    )
    df["year"] = df['date'].map(lambda x: x.year)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_stock_per_pbr(stock_id: int) -> pd.DataFrame:
    api = DataLoader()
    # api.login(user_id='user_id',password='password')
    df = api.taiwan_stock_per_pbr(
        stock_id=F'{stock_id}',
        start_date='2000-01-01',
    )
    return df

def get_mouth_revenue(stock_id: int) -> pd.DataFrame:
    api = DataLoader()
    # api.login(user_id='user_id',password='password')
    df = api.taiwan_stock_month_revenue(
        stock_id=F'{stock_id}',
        start_date='2000-01-01',
    )

    df['date'] = pd.to_datetime(df['date'])
    return df

if __name__ == '__main__':
    print(get_stock_per_pbr(1101))
    pass
