
import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the JSON file
file_path = F'../data/strategy/{input(">")}/data.json'
data = pd.read_json(file_path)

# Define the years range
years = range(1998, 2009)


plt.figure(figsize=(15, 8))


for start_year in years:

  # 填充開始年份前None數據
  returns = [None] * (start_year-1998)  

  try:
    # 填充开始年份後数据
    returns.extend(data[f'SplitBy{start_year}']['YearReturnList']) 

    # 填充结束年份後None數據
    remains = 2009 - start_year - len(returns)
    returns.extend([None] * remains)

  except KeyError:
    pass

  # 畫線 
  plt.plot(years, returns, marker='o', label=f'Start Year: {start_year}')

# 設定圖表
plt.legend()  
plt.xlabel('Year')
plt.ylabel('Return')
plt.title('Yearly Returns for Different Data Sets Starting from Various Years')
plt.grid(True)
plt.show()