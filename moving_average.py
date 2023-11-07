import csv
import os
import glob
import time
from datetime import datetime

# 設定路徑
path = "data/stocks"


stocks = glob.glob(os.path.join(path, "*.csv"))  # 使用 glob 模組找出所有的 csv 檔案
for stock in stocks:
    close = []
    dates = []
    with open(stock, newline="", encoding="utf-8") as csvfile:  # 讀檔
        rows = csv.reader(csvfile)  # 讀取csv檔案
        next(rows, None)  # 跳過第一行
        for row in rows:
            if row[7] != "0.0":  # 取出收盤價 去除0.0
                try:
                    close.append(float(row[7]))  # 將字串轉成浮點數
                    dates.append(
                        datetime.strptime(row[0], "%Y-%m-%d")
                    ) 
                except ValueError:
                    pass  # 如果轉換失敗，忽略該行
    close = close[1:]  # 去除第一行的欄位名稱
    dates = dates[1:]  # 去除第一行的欄位名稱

    # 計算移動平均
    moving_average = []
    for i in range(200, len(close)):
        moving_average.append(sum(close[i - 200 : i]) / 200)

    # 畫圖
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))# 設定x軸主要標籤的格式
    plt.plot(dates, close, label="close")
    plt.plot(dates[200:], moving_average, label="moving average")
    plt.xlabel("year")
    plt.ylabel("stocks price")
    plt.title("Moving Average")
    plt.legend()  # 顯示圖例
    plt.gcf().autofmt_xdate()  # 自動調整 x 軸日期標籤，以避免重疊
    
    if not os.path.exists("data/moving_average"):
        os.mkdir("data/moving_average")
    plt.savefig(f"data/moving_average/{stock[12:16]}.jpg")  # 儲存圖表
    plt.close()  # 關閉圖表
