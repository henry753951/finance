import pandas as pd
import numpy as np
import json

with open(F"../data/strategy/{input('>')}/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print(data)
    print(data.keys())
    data = {

    }
    for key in data.keys():
        data["平均報酬率"] = (np.average(np.array(data[key]["YearReturnList"])))
        data["標準差"] = (np.std(np.array(data[key]["YearReturnList"])))
        data["最大報酬率"] = (np.max(np.array(data[key]["YearReturnList"])))
        data["最小報酬率"] = (np.min(np.array(data[key]["YearReturnList"])))
        data["最大報酬率年份"] = (np.argmax(np.array(data[key]["YearReturnList"])) + 1998)
        data["最小報酬率年份"] = (np.argmin(np.array(data[key]["YearReturnList"])) + 1998)
    
    print(data)
