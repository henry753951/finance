import requests
import pandas as pd
import os,datetime
stockID = input("請輸入股票代號：")
url = "https://api.finmindtrade.com/api/v4/data"
last_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
if F"{stockID}.csv" in os.listdir("data/news"):
    data = pd.read_csv(f"data/news/{stockID}.csv")
    last_date = data.iloc[-1]['date']
    last_date = last_date.split(" ")[0]

parameter = {
    "dataset": "TaiwanStockNews",
    "data_id":stockID,
    "start_date": last_date,
    "end_date": datetime.datetime.now().strftime("%Y-%m-%d"),
}
data = requests.get(url, params=parameter)
data = data.json()
data = pd.DataFrame(data['data'])
if F"{stockID}.csv" in os.listdir("data/news"):
    data = pd.concat([data, pd.read_csv(f"data/news/{stockID}.csv")])
else:
    data.to_csv(f"data/news/{stockID}.csv", index=False)
