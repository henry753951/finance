import requests
from bs4 import BeautifulSoup
import pandas as pd

url='https://mops.twse.com.tw/mops/web/t51sb02'



form_data = {
    'encodeURIComponent':'1',
    'step':'1',
    'firstin':'1',
    'off':'1',
    'TYPEK':'sii',
    'year':'81',
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
    'CRR (%)'  # Cash Reinvestment Ratio
]
df.columns = columns
print(df)