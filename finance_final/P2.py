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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

# Plotting graphs
import matplotlib.pyplot as plt
import json

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
col_name = list(permutations(col_name, 4))

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
            # 將數據分割為訓練集和測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )

            # 建立Decision Tree模型
            dt = DecisionTreeClassifier()

            # 訓練模型
            dt.fit(X_train, y_train)

            # 預測測試集
            y_pred = dt.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)
            temp_accuracy.append(accuracy)
        except Exception as e:
            if e == KeyboardInterrupt:
                raise
            continue

    accuracy_list.append(temp_accuracy[-1])  # 取最後一年的準確率

    if temp_accuracy[-1] > best_accuracy:
        best_accuracy = temp_accuracy[-1]
        best_col = col

    print(accuracy_list)
    result = [
        {"cols": col_name[i], "accuracy": accuracy_list[i]}
        for i in range(len(accuracy_list))
    ]

    # write to json

    with open("P2result.json", "w", encoding="utf8") as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=4)
print(best_accuracy, best_col)

