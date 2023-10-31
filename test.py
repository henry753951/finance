import time
from bs4 import BeautifulSoup
import yfinance as yf
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.150 Safari/537.36"
}


def getHistoryData(id: str):
    try:
        stock = yf.Ticker(id)
        stockData = stock.history(period="Max" )
        return stockData
    except:
        return None




def getHistoryEquityPreYear(stock_id: str):
    try:
        refer = {
            "Referer": F"https://goodinfo.tw/tw/StockAssetsStatus.asp?STOCK_ID={stock_id}"
        }
        response = requests.post(F"https://goodinfo.tw/tw/StockAssetsStatus.asp?STOCK_ID={stock_id}", headers={**headers, **refer})
        soup = BeautifulSoup(response.text, "html.parser")
        soup = soup.find("table", {"id": "tblDetail"})
        rows = soup.find_all("tr")
        data = {}
        for row in rows:
            try:
                cols = row.find_all("td")
                year = cols[0].text if 'Q' not in cols[0].text else "20" + cols[0].text.split("Q")[0]
                equity = int(str(cols[1].text.replace(",","")))
            except:
                continue
            data[year] = equity
        return data
    except Exception as e:
        print(e)
        return None
        
     

def main(stock_id = "2330"):
    df = getHistoryData(stock_id + ".TW")
    if df is None:
        print(F"{stock_id} is None")
        return
    equity = getHistoryEquityPreYear(stock_id)
    if equity is None:
        return

    dataLs=[]
    for index, row in df.iterrows():
        data = {
            "date":str(row.name),
            "close":row["Close"],
            # "open":row["Open"],
            # "high":row["High"],
            # "low":row["Low"],
            # "volume":row["Volume"],
            # "dividends":row["Dividends"],
            # "stock splits":row["Stock Splits"],
            "equity":equity[str(row.name)[0:4]]*row["Close"] if  str(row.name)[0:4] in equity else None
        }
        dataLs.append(data)

    # out csv
    import pandas as pd
    df = pd.DataFrame(dataLs)
    df.to_csv(F"data/{stock_id}TW.csv", index=False)
    

if __name__ == "__main__":
    for id in range(1100,10000):
        main(str(id))
        time.sleep(4)
