import pandas as pd
import utils
from crawl import goodinfo
from datetime import datetime as dt

from finlab import data
import pandas as pd


def main():
    stocks = utils.get_all_stock(3715, 4000)
    market_value_df = data.get("etl:market_value")
    print("股票總數:" + str(len(stocks)))
    for stock in stocks:
        print(stock.stock_id)

        ## 拿舊的資料
        try:
            df = pd.read_csv(f'data/stocks/{stock.stock_id}.csv')
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].map(lambda x: x.year)

            print("use old data")
        except FileNotFoundError:
            df = utils.get_stock_daily(int(stock.stock_id))

            try:
                if df['date'] is None:
                    continue
            except:
                continue

        ## PER PBR 本益比 淨值比
        if 'PER' not in df.columns:
            tempDF = utils.get_stock_per_pbr(int(stock.stock_id))
            if tempDF is None:
                df['PER'] = 0
                df['PBR'] = 0
            else:
                tempDF = tempDF[['date', 'PER', 'PBR']]
                df = pd.merge(df, tempDF, on='date', how='inner')

        ## 市值
        if 'equity' not in df.columns or df['equity'].sum() == 0:
            equity = None
            equity = goodinfo.getHistoryEquityPreYear(stock.stock_id)
            if equity is not None:
                df["equity"] = df['year'].map(lambda x: equity[str(x)] if str(x) in equity else 0)
                df['mkPrice'] = df['close'] * df['equity']
            else:
                try:
                    market_value_df = market_value_df["date", str(stock.stock_id)]
                    market_value_df = market_value_df.rename(
                        columns={str(stock.stock_id): "market_value"}
                    )
                    market_value_df["date"] = pd.to_datetime(market_value_df["date"])
                    df = pd.merge(df, market_value_df, on="date", how="inner")
                except Exception as e:
                    df["equity"] = 0
                    df["mkPrice"] = 0

        ## 股價營收比 (市值 / 年營收) PSR
        if 'mkPrice' in df.columns:
            mouth_revenue = utils.get_mouth_revenue(int(stock.stock_id))

            if mouth_revenue is None:
                df['PSR'] = 0
            else:
                mouth_revenue = mouth_revenue.sort_values('date', ascending=False)
                for index, row in df.iterrows():
                    date = row['date']
                    previous_12_records = mouth_revenue[mouth_revenue['date'] < date].head(12)
                    sum = previous_12_records['revenue'].sum()
                    df.at[index, "PSR"] = row['mkPrice']*10000000 / sum if sum != 0 else 0

            
        ## ROA ROE
        if 'ROA' not in df.columns and 'ROE' not in df.columns:
            try:
                ROAEdf = pd.read_csv(f'data/roa_roe/single_company_history/company_{stock.stock_id}.csv',usecols=['年份','ROA','ROE'])
                ROAEdf = ROAEdf.rename(columns={'年份': 'year'})
                df = pd.merge(df, ROAEdf, on='year', how='inner')
            except Exception as e:
                df['ROA'] = 0
                df['ROE'] = 0
                print("no ROA ROE data")

        df.to_csv(f'data/stocks/{stock.stock_id}.csv', index=False)


if __name__ == '__main__':
    main()
