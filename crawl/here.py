from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
import time
import random
import pandas as pd
import re
import os
import shutil

df=pd.read_csv('taiwan_stock_info.csv')
stock_name= df.query("(stock_id>'1100'&stock_id<'9999')")
stok_id=df[['stock_id', 'stock_name']]
#print(stok_id)
new_df = stock_name.loc[(df['stock_id'].astype(str).str.len() < 5) & (~df['stock_id'].astype(str).str.contains('[a-zA-Z]')), 'stock_name']

################
driver = webdriver.Chrome()
driver.get('https://trends.google.com/trends/?geo=TW')
element = driver.find_element(By.XPATH,'.//*[@id="i7"]')
driver.maximize_window()
keywords = list(new_df)
print(new_df)
for key in keywords: 
#loop through the keywords
    valid_filename = re.sub(r'[\\/*?:"<>|]', '_', key)
    if os.path.exists(f'data/{valid_filename}.csv'):
        print(f'{key}.csv already exists. Skipping...')
        continue
    driver.close()
    driver = webdriver.Chrome()
    driver.get('https://trends.google.com/trends/?geo=TW')
    element = driver.find_element(By.XPATH,'.//*[@id="i7"]')
    driver.maximize_window()
    element.clear()
    time.sleep(1)
    element.clear()
    element.send_keys(key)
    element.send_keys(Keys.ENTER)
    #_md-select-icon
    try:
        WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div[2]/div/header/div/div[3]/ng-transclude/div[2]/div/div/custom-date-picker/ng-include/md-select/md-select-value/span[1]/div"))).click()
        time.sleep(1)
        elementbtn = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="select_option_17"]/div')))
        driver.execute_script("arguments[0].click();", elementbtn)
        time.sleep(1)
        WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div[2]/div/md-content/div/div/div[1]/trends-widget/ng-include/widget/div/div/div/widget-actions/div/button[1]"))).click()
    except StaleElementReferenceException as se:
        print("StaleElementReferenceException: Element reference is stale or no longer valid")
        continue
    except TimeoutException as te:
        print("TimeoutException: Element not found or not clickable")
        continue
    except Exception as e:
        print("Exception occurred:", str(e))
        continue
    time.sleep(1)
    # 原始檔案路徑
    original_path = "C:\\Users\\snoopy\\Downloads\\multiTimeline.csv"

    # 取得有效的檔名
    

    # 新的檔案路徑
    new_path = os.path.join(os.path.dirname(original_path), f"{valid_filename}.csv")

    # 重新命名檔案
    os.rename(original_path, new_path)
    
    shutil.move(new_path, 'data')
    clock=random.randint(5,12)
    time.sleep(clock)
    