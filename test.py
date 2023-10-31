from bs4 import BeautifulSoup
import yfinance as yf
import requests

url = "https://tw.stock.yahoo.com/quote/2330.TW/"
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
    rows = [row.find("td").text for row in rows]
    for row in rows:
        try:
            print(row[1])
            print(int(str(row[1]).replace(",","")))
        except ValueError:
            continue


def main():
    stock_id = "2330"
    getHistoryEquityPreYear(stock_id)


if __name__ == "__main__":
    main()
