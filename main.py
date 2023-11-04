import pandas as pd
import utils
from crawl import goodinfo
from datetime import datetime as dt


def main():
    stocks = utils.get_all_stock(1101, 5000)
    print(len(stocks))
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
            if df['date'] is None:
                continue

        ## PER PBR 本益比 淨值比
        if 'PER' not in df.columns:
            tempDF = utils.get_stock_per_pbr(int(stock.stock_id))
            tempDF = tempDF[['date', 'PER', 'PBR']]
            df = pd.merge(df, tempDF, on='date', how='inner')

        ## 市值
        equity = None
        equity = goodinfo.getHistoryEquityPreYear(stock.stock_id)
        print(equity)
        if equity is not None:
            df["equity"] = df['year'].map(lambda x: equity[str(x)] if str(x) in equity else 0)
            df['mkPrice'] = df['close'] * df['equity']

        ## 股價營收比 (市值 / 年營收) PSR
        if equity is not None:
            mouth_revenue = utils.get_mouth_revenue(int(stock.stock_id))
            mouth_revenue = mouth_revenue.sort_values('date', ascending=False)
            for _, row in df.iterrows():
                date = row['date']
                previous_12_records = mouth_revenue[mouth_revenue['date'] < date].head(12)
                sum = previous_12_records['revenue'].sum()
                row["PSR"] = row['mkPrice'] / sum if sum != 0 else 0

        df.to_csv(f'data/stocks/{stock.stock_id}.csv', index=False)


if __name__ == '__main__':
    main()
