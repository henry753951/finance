from DataLoader import DataLoader

# Data Manipulation
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

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from itertools import permutations

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


best_accuracy = -1
best_col = []

all = pd.read_csv("../data/top200_training.csv")
# 去除空白
for col in all.columns:
    all[col] = all[col].astype(str)
    all[col] = all[col].str.strip()


accuracy_list = []
col_name = [
    col
    for col in all.columns
    if col
    not in [
        "證券代碼",
        "年月",
        "簡稱",
    ]
]  # 取出所有欄位名稱
# 排列組合col_name
col_name = list(permutations(col_name, 6))

stocks = list(set(all["證券代碼"].to_list()))  # 取出所有股票代碼

for col in col_name:
    temp_accuracy = []
    for year in range(1997, 2010):
        try:
            X = pd.DataFrame()
            Y = np.array([])
            print(year)
            for stock in stocks:
                df = all[all["證券代碼"] == stock]
                # 只保留df["年月"]的年份
                df.loc[:, "年月"] = df["年月"].astype(str).str.strip().str[:4].astype(int)
                df.set_index("年月", inplace=True)
                df.sort_index(inplace=True)

                dfreg = df.loc[:, list(col)]
                dfreg.dropna(inplace=True)
                X = pd.concat([X, dfreg])  # 連接資料
                Y_values = df["ReturnMean_year_Label"].shift(-1)
                Y_values.dropna(inplace=True)
                X = X[:-1]
                Y = np.concatenate([Y, Y_values.values])

            cv.fit(X, Y)
            print(cv.best_params_["n_neighbors"])

            knn = KNeighborsClassifier(n_neighbors=cv.best_params_["n_neighbors"])
            X_train, X_test, Y_train, Y_test = SplitData(X, Y, year).split_data()
            if X_train.empty or len(X_train) < cv.best_params_["n_neighbors"]:
                continue
            # 訓練模型
            knn.fit(X_train, Y_train)
            # 預測
            pred = knn.predict(X_test)
            accuracy = accuracy_score(Y_test, pred)
            temp_accuracy.append(accuracy)
        except:
            continue

        # # 交易信號
        # trade = pd.DataFrame()
        # trade["Predict_Signal"] = knn.predict(X)
        # # AAPL Cumulative Returns
        # trade["收盤價(元)_年"] = trade["收盤價(元)_年"].astype(float)
        # trade["SPY_data_returns"] = np.log(
        #     df["收盤價(元)_年"] / trade["收盤價(元)_年"].shift(1)
        # )  # 計算收益率
        # Cumulative_SPY_data_returns = trade[["SPY_data_returns"]].cumsum() * 100  # 計算累積收益率

        # trade["Startegy_returns"] = trade["SPY_data_returns"] * trade[
        #     "Predict_Signal"
        # ].shift(
        #     1
        # )  # 計算策略收益率
        # Cumulative_Strategy_returns = (
        #     trade[["Startegy_returns"]].cumsum() * 100
        # )  # 計算累積策略收益率

        # Plot the results to visualize the performance
        # plt.figure(figsize=(10, 5))
        # plt.plot(Cumulative_SPY_data_returns, color="r", label="SPY Returns")
        # plt.plot(Cumulative_Strategy_returns, color="g", label="Strategy Returns")
        # plt.legend()
        # plt.show()

    accuracy_list.append(temp_accuracy[-1])  # 取最後一年的準確率

    if temp_accuracy[-1] > best_accuracy:
        best_accuracy = temp_accuracy[-1]
        best_col = col


result = [
    {"cols": col_name[i], "accuracy": accuracy_list[i]}
    for i in range(len(accuracy_list))
]

# write to json
import json

with open("result.json", "w") as outfile:
    json.dump(result, outfile)
