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
    print(year)
    X = pd.DataFrame()
    Y = np.array([])

    for stock in stocks:
        df = all[all["證券代碼"] == stock]
        # 只保留df["年月"]的年份
        df.loc[:, "年月"] = df["年月"].astype(str).str.strip().str[:4].astype(int)
        df.set_index("年月", inplace=True)
        df.sort_index(inplace=True)

        dfreg = df.loc[:, col_name]
        dfreg.dropna(inplace=True)
        X = pd.concat([X, dfreg])  # 連接資料
        Y_values = df["ReturnMean_year_Label"].shift(-1)
        Y_values.dropna(inplace=True)
        X = X[:-1]
        Y = np.concatenate([Y, Y_values.values])

    X_train, X_test, Y_train, Y_test = SplitData(X, Y, year).split_data()
    if X_train.empty:
        continue

    # 特徵選取
    rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    rf.fit(X_train, Y_train)
    # print("特徵重要程度: ", rf.feature_importances_)
    feature_index = np.array(rf.feature_importances_.argsort()[-10:][::-1])  # 取前10個重要特徵
    feature_index.sort()
    new_feature = [X.columns[i] for i in feature_index]
    print("特徵: ", new_feature)
    # print("feature_index", feature_index)
    X_train = X_train.loc[:][new_feature]
    X_test = X_test.loc[:][new_feature]

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

    probs = knn.predict_proba(X_test)[:, 1]
    df = pd.DataFrame({"證券代碼": stock, "預測報酬率": probs})
    df.set_index("證券代碼", inplace=True)

    predict_df = pd.concat([predict_df, df])
    predict_df = (
        predict_df.sort_values("預測報酬率", ascending=False)
        .groupby(predict_df.index)
        .head(1)
    )  # 取出每年預測報酬率最高的股票
    print(predict_df)

print("平均準確率: ", np.mean(accuracy_list))
