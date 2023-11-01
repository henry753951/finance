import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://mops.twse.com.tw/mops/web/t51sb02'


stocks = {}

for y in range(78, 101):
    year = 1911 + y
    form_data = {
        'encodeURIComponent': '1',
        'step': '1',
        'firstin': '1',
        'off': '1',
        'TYPEK': 'sii',
        'year': F'{y}',
    }
    r = requests.post(url, data=form_data)
    df = pd.read_html(r.text)[10]

    columns = [
        'stock_id',
        'Company Name',
        'DAR (%)',  # Debt to Asset Ratio
        'LFAR (%)',  # Long-term Funds to Fixed Assets Ratio
        'CR (%)',  # Current Ratio
        'QR (%)',  # Quick Ratio
        'ICR (%)',  # Interest Coverage Ratio
        'ART',  # Accounts Receivable Turnover
        'DSO',  # Days Sales Outstanding
        'ITR',  # Inventory Turnover Ratio
        'DSI',  # Days Sales of Inventory
        'FATR',  # Fixed Assets Turnover Ratio
        'TATR',  # Total Assets Turnover Ratio
        'ROA (%)',  # Return on Assets
        'ROE (%)',  # Return on Equity
        'OPPCR (%)',  # Operating Profit to Paid-in Capital Ratio
        'NPCR (%)',  # Pre-tax Net Income to Paid-in Capital Ratio
        'NPM (%)',  # Net Profit Margin
        'EPS',  # Earnings per Share
        'CFR (%)',  # Cash Flow Ratio
        'CFAR (%)',  # Cash Flow Adequacy Ratio
        'CRR (%)',  # Cash Reinvestment Ratio
    ]
    df.columns = columns

    df["year"] = year
    df = df[df['stock_id'].apply(lambda x: str(x).isdigit())]


    for _, row in df.iterrows():
        stock_id = row['stock_id']
        if stock_id not in stocks:
            stocks[stock_id] = pd.DataFrame(columns=["year", "ROA (%)"])
        stocks[stock_id] = stocks[stock_id].append(row[['year', 'ROA (%)']])
        stocks[stock_id].to_csv(f'data/roa/{stock_id}.csv', index=False)

    