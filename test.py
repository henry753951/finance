from bs4 import BeautifulSoup
import yfinance as yf
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.150 Safari/537.36"
}


def getHistoryData(id: str):
    stock = yf.Ticker(id)
    stockData = stock.history(period="max")
    return stockData




def getHistoryEquityPreYear(stock_id: str):
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
        
     

def main():
    stock_id = "2330"
    # getHistoryEquityPreYear(stock_id)
    df = getHistoryData(stock_id)
    equity = getHistoryEquityPreYear(stock_id)
    dataLs=[]
    for index, row in df.iterrows():
        data = {
            "date":str(row.name),
            "close":row["Close"],
            "open":row["Open"],
            "high":row["High"],
            "low":row["Low"],
            "volume":row["Volume"],
            "dividends":row["Dividends"],
            "stock splits":row["Stock Splits"],
            "equity":equity[str(row.name)[0:4]]*row["Close"]
        }
        dataLs.append(data)

        
    print(dataLs[0])
    print(equity["2002"])

if __name__ == "__main__":
    main()
