import pprint

from sklearn.svm import SVC
from DataLoader import DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from itertools import permutations
import json
from sklearn.ensemble import RandomForestClassifier
import os

import utils

n_neighbors = list(range(1, 11, 1))
cv = GridSearchCV(
    estimator=KNeighborsClassifier(), param_grid={"n_neighbors": n_neighbors}, cv=5
)


class SplitData:
    def __init__(self, X: pd.DataFrame, Y: np.ndarray, year: int):
        self.X = X
        self.Y = Y
        self.year = year

    def split_data(self):
        X_train = self.X[self.X.index < self.year]
        X_test = self.X[self.X.index >= self.year]
        Y_train = self.Y[: len(X_train)]
        Y_test = self.Y[len(X_train) :]
        return X_train, X_test, Y_train, Y_test


if not os.path.exists(f"../data/strategy/4/"):
    os.makedirs(f"../data/strategy/4/")
output: list[dict] = []


def getStockByYear(year, stock_id) -> dict or bool:
    try:
        find: pd.DataFrame = all_indexed.loc[(int(stock_id), year)]
    except KeyError:
        return False
    return find.to_dict()


def strategy(Y_pred: np.ndarray, X_test_copy: pd.DataFrame):
    # X_test_strategy = 2000 - 2005
    # Y_pred = 2001 - 2006

    # for loop skip first year
    # last year unknown
    for index, row in enumerate(X_test_copy.tail(len(Y_pred)).iterrows()):
        if Y_pred[index] == "1":
            NextYearStockData = getStockByYear(row[0] + 1, row[1]["證券代碼"])
            if not NextYearStockData:
                continue
            print(row[0] + 1, row[1]["簡稱"])
            output.append(
                {
                    "year": row[0] + 1,
                    "stock": row[1]["證券代碼"],
                    "stock_name": row[1]["簡稱"],
                    "open_price": float(row[1]["收盤價(元)_年"]),
                    "close_price": float(NextYearStockData["收盤價(元)_年"]),
                    "return": float(NextYearStockData["收盤價(元)_年"])
                    / float(row[1]["收盤價(元)_年"]),
                }
            )


dl = DataLoader()
dl.load_stocks(cache=True)
df = dl.getAllStocksPreYear(
    cols=["equity", "Trading_turnover"], cols_deal=[None, utils.calcTurnover]
)
not_exist = []

accuracy_list = []

all = pd.read_csv("../data/top200_training.csv")
all_indexed = all.copy()
all_indexed.loc[:, "年月"] = all_indexed["年月"].astype(str).str.strip().str[:4].astype(int)
all_indexed.loc[:, "證券代碼"] = all_indexed["證券代碼"].astype(str)

all_indexed.set_index(["證券代碼", "年月"], inplace=True)

# 去除空白
for col in all:
    all[col] = all[col].astype(str)
    all[col] = all[col].str.strip()

# 去除不要的欄位
col_name = [
    col
    for col in all.columns
    if col
    not in [
        "證券代碼",
        "年月",
        "簡稱",
    ]
]
df = pd.merge(
    all_indexed,
    df,
    left_on=["證券代碼", "年月"],
    right_on=["stock_id", "year"],
    how="inner",
)
stocks = df["stock_id"].unique().tolist()
for year in range(1998, 2009):
    output = []
    print(f"Year = {year}")
    X = pd.DataFrame()
    Y = np.array([])

    for stock in stocks:
        df = all[all["證券代碼"] == stock]
        # 只保留df["年月"]的年份
        df.loc[:, "年月"] = df["年月"].astype(str).str.strip().str[:4].astype(int)
        df.set_index("年月", inplace=True)
        df = df.dropna(inplace=False)
        X = pd.concat([X, df])  # 連接資料
        Y_values = df["ReturnMean_year_Label"].shift(-1)
        Y_values.dropna(inplace=True)
        X = X[:-1]
        Y = np.concatenate([Y, Y_values.values])

    X = X.sort_index()
    X_train, X_test, Y_train, Y_test = SplitData(X, Y, year).split_data()

    print("X_test: ", len(X_test), "Y_test: ", len(Y_test))
    if X_train.empty:
        continue

    # 特徵選取
    rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    rf.fit(X_train.loc[:, col_name], Y_train)
    print("特徵重要程度: ", rf.feature_importances_)
    feature_index = np.array(rf.feature_importances_.argsort()[-10:][::-1])  # 取前10個重要特徵
    feature_index.sort()
    new_feature = [X.loc[:, col_name].columns[i] for i in feature_index]
    print("特徵: ", new_feature)
    # print("feature_index", feature_index)
    X_test_strategy = X_test.copy()
    X_train = X_train.loc[:, col_name].loc[:][new_feature]
    X_test = X_test.loc[:, col_name].loc[:][new_feature]
    # 搜尋最佳K值
    cv.fit(X_train, Y_train)
    print("K值: ", cv.best_params_["n_neighbors"])

    # 訓練模型

    param_grid = {
        "C": [1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.001, 0.0001],
        "kernel": ["linear", "rbf"],
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    # 預測
    grid.fit(X_train, Y_train)
    Y_pred = grid.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("準確率: ", accuracy)
    accuracy_list.append(accuracy)
    # 策略
    strategyMoney = strategy(Y_pred, X_test_strategy)
    print("策略: ", strategyMoney)
    print(f"Count_1: {list(Y_pred).count('1')} Count_-1: {list(Y_pred).count('-1')}")

    # out csv

    output_df = pd.DataFrame(output)
    output_df.sort_values(by=["return"], ascending=False, inplace=True)  # 排序

    perStockMoney = 10
    YearReturnList = []
    year_list = list(set(output_df["year"].to_list()))
    for _year in year_list:
        temp_df = output_df.copy()
        year_df = temp_df[temp_df["year"] == _year]
        stock_count = len(year_df)
        sum_ = year_df["return"].sum()
        YearReturn = sum_ / stock_count
        print(f"{_year}年報酬率: {YearReturn}")
        YearReturnList.append(YearReturn)

    # 計算年報酬率
    output_df.to_csv(f"../data/strategy/1/SplitBy{year}.csv", index=False)
    if "data.json" not in os.listdir("../data/strategy/1/"):
        with open(f"../data/strategy/1/data.json", "w", encoding="utf-8") as outfile:
            json.dump(
                {},
                outfile,
                ensure_ascii=False,
            )
    with open(f"../data/strategy/1/data.json", "r", encoding="utf-8") as outfile:
        data = json.load(outfile)
        data[f"SplitBy{year}"] = {
            "accuracy": accuracy,
            "YearReturnList": YearReturnList,
            "特徵": new_feature,
            "特徵重要程度": sorted(
                list(rf.feature_importances_.argsort()[-10:][::-1].astype(float))
            ),
            "K值": cv.best_params_["n_neighbors"],
        }
    with open(f"../data/strategy/1/data.json", "w", encoding="utf-8") as outfile:
        json.dump(
            data,
            outfile,
            ensure_ascii=False,
            indent=4,
        )
