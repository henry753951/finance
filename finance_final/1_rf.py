import pprint
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

n_neighbors = list(range(1, 11, 1))  # 設定K值範圍
cv = GridSearchCV(
    estimator=KNeighborsClassifier(), param_grid={"n_neighbors": n_neighbors}, cv=5
)  # 搜尋最佳K值


# 拆分資料
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


all = pd.read_csv("../data/top200_training.csv")


def strategy(Y_pred: np.ndarray, X_test_copy: pd.DataFrame):
    strategyMoney = 1000000
    stocksIhav = {}
    currentYear = None
    buylist = []
    selllist = []
    for index, row in enumerate(X_test_copy.tail(len(Y_pred)).iterrows()):
        if currentYear != row[0]:  # 換年
            # buy
            currentYear = row[0]
            for stock, data in stocksIhav.items():
                strategyMoney += data["count"] * data["price"]

        if Y_pred[index] == "1":
            buylist.append(row[1]["證券代碼"])
            # print(f"buy {row[1]['證券代碼']}")
            stocksIhav[row[1]["證券代碼"]] = {
                "count": (strategyMoney / len(Y_pred)) // float(row[1]["收盤價(元)_年"]),
                "price": float(row[1]["收盤價(元)_年"]),
            }
        elif Y_pred[index] == "-1":
            # sell
            selllist.append(row[1]["證券代碼"])
            # print(f"sell {row[1]['證券代碼']}")
            if row[1]["證券代碼"] in stocksIhav:
                strategyMoney -= stocksIhav[row[1]["證券代碼"]]["count"] * float(
                    row[1]["收盤價(元)_年"]
                )
                stocksIhav.pop(row[1]["證券代碼"])
    for stock, data in stocksIhav.items():
        strategyMoney += data["count"] * data["price"]  # 最後一年賣掉
    print("buylist: ", buylist)
    print("selllist: ", selllist)
    buylist = []
    selllist = []
    return strategyMoney


# 去除空白
for col in all.columns:
    all[col] = all[col].astype(str)
    all[col] = all[col].str.strip()

accuracy_list = []
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

stocks = list(set(all["證券代碼"].to_list()))  # 取出所有股票代碼
predict_df = pd.DataFrame()
for year in range(1998, 2009):
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
    # print("特徵重要程度: ", rf.feature_importances_)
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

    knn = KNeighborsClassifier(n_neighbors=cv.best_params_["n_neighbors"])
    knn.fit(X_train, Y_train)
    # 預測

    Y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("準確率: ", accuracy)
    accuracy_list.append(accuracy)
    # 策略
    strategyMoney = strategy(Y_pred, X_test_strategy)
    print("策略: ", strategyMoney)
    print(f"Count_1: {list(Y_pred).count('1')} Count_-1: {list(Y_pred).count('-1')}")
