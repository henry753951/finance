from sklearn.svm import SVC
from DataLoader import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV as rcv
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json
from sklearn.ensemble import RandomForestClassifier
import os

import utils

n_neighbors = list(range(1, 11, 1))
cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={"n_neighbors": n_neighbors}, cv=5)


if not os.path.exists(f"../data/strategy/4/"):
    os.makedirs(f"../data/strategy/4/")
output: list[dict] = []


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
for col in ["簡稱", "證券代碼", "年月"]:
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
all = pd.merge(
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

    X_train, X_test, Y_train, Y_test = utils.SplitData(utils.normalization(X), Y, year).split_data()
    _, X_test_UNnormalized, _, _ = utils.SplitData(X, Y, year).split_data()
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
    X_train = X_train.loc[:, col_name].loc[:][new_feature]
    X_test = X_test.loc[:, col_name].loc[:][new_feature]
    # 搜尋最佳K值
    cv.fit(X_train, Y_train)
    print("K值: ", cv.best_params_["n_neighbors"])

    # 訓練模型
    print(X_train)
    print(X_test.shape)
    print("====")
    param_grid = {
        "C": [1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.001, 0.0001],
        "kernel": ["linear", "rbf"],
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=10)
    # 預測
    grid.fit(X_train, Y_train)
    Y_pred = grid.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("準確率: ", accuracy)
    accuracy_list.append(accuracy)
    # 策略
    print("策略")
    output += utils.strategy(Y_pred, X_test_UNnormalized, all_indexed)
    print(output)
    print(f"Count_1: {list(Y_pred).count(1)} Count_-1: {list(Y_pred).count(-1)}")

    YearReturnList = []
    # out csv
    if len(output) != 0:
        output_df = pd.DataFrame(output)
        output_df.sort_values(by=["return"], ascending=False, inplace=True)  # 排序

        perStockMoney = 10
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
    output_df.to_csv(f"../data/strategy/4/SplitBy{year}.csv", index=False)
    if "data.json" not in os.listdir("../data/strategy/4/"):
        with open(f"../data/strategy/4/data.json", "w", encoding="utf-8") as outfile:
            json.dump(
                {},
                outfile,
                ensure_ascii=False,
            )
    with open(f"../data/strategy/4/data.json", "r", encoding="utf-8") as outfile:
        data = json.load(outfile)
        data[f"SplitBy{year}"] = {
            "accuracy": accuracy,
            "YearReturnList": YearReturnList,
            "特徵": new_feature,
            "特徵重要程度": sorted(list(rf.feature_importances_.argsort()[-10:][::-1].astype(float))),
            "K值": cv.best_params_["n_neighbors"],
        }
    with open(f"../data/strategy/4/data.json", "w", encoding="utf-8") as outfile:
        json.dump(
            data,
            outfile,
            ensure_ascii=False,
            indent=4,
        )
