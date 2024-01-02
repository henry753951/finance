import pandas as pd
import numpy as np
import json

Part = input('>')
dataList = []
with open(F"../data/strategy/{Part}/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)


for key in data.keys():
    temp = {}
    print(np.average(np.array(data[key]["YearReturnList"])))
    temp["平均報酬率"] = np.average(np.array(data[key]["YearReturnList"]))
    temp["標準差"] = np.std(np.array(data[key]["YearReturnList"]))
    temp["最大報酬率"] = np.max(np.array(data[key]["YearReturnList"]))
    temp["最小報酬率"] = np.min(np.array(data[key]["YearReturnList"]))
    temp["最大報酬率年份"] = np.argmax(np.array(data[key]["YearReturnList"])) + 1998
    temp["最小報酬率年份"] = np.argmin(np.array(data[key]["YearReturnList"])) + 1998
    temp["準確率"] = np.average(np.array(data[key]["accuracy"]))
    dataList.append(temp)

pd.DataFrame(dataList).to_csv(F"../data/strategy/{Part}/data.csv", index=False)
