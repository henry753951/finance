from bs4 import BeautifulSoup
import yfinance as yf
import httpx

url = "https://tw.stock.yahoo.com/quote/2330.TW/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.150 Safari/537.36"
}


def getHistoryData(id: str):
    stock = yf.Ticker(id)
    stockData = stock.history(period="max")
    return stockData


async def getHistoryEquityPreYear():
    response = httpx.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
