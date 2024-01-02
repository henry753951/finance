import pandas as pd
import numpy as np
import json

with open(F"../data/strategy/{input('>')}/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print(data)
    print(data.keys())
    for key in data.keys():
        print(key)
        print(data[key]["accuracy"])
        print(data[key]["YearReturnList"])
