import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


    
def year_comparion():
    for y in range(78, 101):
        form_data = {
        'encodeURIComponent':'1',
        'step':'1',
        'firstin':'1',
        'off':'1',
        'TYPEK':'sii',
        'year':F'{y}',
        }
        url='https://mops.twse.com.tw/mops/web/ajax_t51sb02'
        r2 = requests.post(url, data=form_data)
        soup = BeautifulSoup(r2.text, 'html.parser')
        #print (soup)
        first= soup.find_all('table',class_='hasBorder')
        print(first)
        even = first[0].find_all('tr', class_='even')
        odd = first[0].find_all('tr', class_='odd')

        data = []

        for stock in even:
            company_code = stock.find_all('td')[0].text
            company_name = stock.find_all('td')[1].text
            roe = stock.find_all('td')[13].text
            roa = stock.find_all('td')[14].text
            data.append([company_code, company_name, roe, roa])
    
        for stock in odd:
            company_code = stock.find_all('td')[0].text
            company_name = stock.find_all('td')[1].text
            roe = stock.find_all('td')[13].text
            roa = stock.find_all('td')[14].text
            data.append([company_code, company_name, roe, roa])

        df = pd.DataFrame(data, columns=['公司代號', '公司名稱', 'ROE', 'ROA'])
        df=df.sort_values(by='公司代號')
        filename = f'stocks_{y+1911}.csv'
        path='./year_comparison/'+filename
        df.to_csv(path, index=False)
    ############################################

    for y in range(102, 112):
        form_data_new = {
            'encodeURIComponent': '1',
            'run': 'Y',
            'step': '1',
            'TYPEK': 'sii',
            'year': F'{y}',
            'isnew': '',
            'firstin': '1',
            'off': '1',
            'ifrs': 'Y',
        }
        url_new = 'https://mops.twse.com.tw/mops/web/ajax_t51sb02'
        r2 = requests.post(url_new, data=form_data_new)
        soup = BeautifulSoup(r2.text, 'html.parser')
        first= soup.find_all('table',class_='hasBorder')
        even = first[0].find_all('tr', class_='even')
        odd = first[0].find_all('tr', class_='odd')

        data = []

        for stock in even:
            company_code = stock.find_all('td')[0].text
            company_name = stock.find_all('td')[1].text
            roe = stock.find_all('td')[13].text
            roa = stock.find_all('td')[14].text
            data.append([company_code, company_name, roe, roa])
    
        for stock in odd:
            company_code = stock.find_all('td')[0].text
            company_name = stock.find_all('td')[1].text
            roe = stock.find_all('td')[13].text
            roa = stock.find_all('td')[14].text
            data.append([company_code, company_name, roe, roa])

        df = pd.DataFrame(data, columns=['公司代號', '公司名稱', 'ROE', 'ROA'])
        df=df.sort_values(by='公司代號')
        filename = f'stocks_{y+1911}.csv'
        path='year_comparison/'+filename
        df.to_csv(path, index=False)
    
#分隔線是我------------------------------------------------------------------------------------------

def single_company_history():
    
# 建立一個空的DataFrame用於存儲合併後的資料
    merged_data = pd.DataFrame()

# 讀取年份範圍內的CSV檔案
    for year in range(1989, 2023):
        filename = './year_comparison/'+F'stocks_{year}.csv'
        try:
            df = pd.read_csv(filename)
            df.insert(0, '年份', year)
            print(df)
        # 將當前年份的資料與已合併的資料進行合併
            merged_data = pd.concat([merged_data, df], ignore_index=True)
        except FileNotFoundError:
            print(F"找不到檔案 {filename}")

# 以'公司代號'為依據，將資料分組
    grouped = merged_data.groupby('公司代號')
# 逐個公司代號建立CSV檔案
    for company_code, group in grouped:
        company_filename = f'company_{company_code}.csv'
        path='./single_company_history/'+company_filename
        group.to_csv(path, index=False)
        #print(f"已創建檔案 {company_filename}")
        
        
single_company_history()