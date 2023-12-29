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
        X_train = self.X[self.X.index < str(self.year)]
        X_test = self.X[self.X.index >= str(self.year)]
        Y_train = self.Y[self.X.index < str(self.year)]
        Y_test = self.Y[self.X.index >= str(self.year)]
        return X_train, X_test, Y_train, Y_test


data = pd.read_csv("../data/top200_training.csv")
# 去除空白
for col in data.columns:
    data[col] = data[col].astype(str)
    data[col] = data[col].str.strip()

stock_id = "2330"
accuracy_list = []

for year in range(1998, 2010):
    print(year)
    df = data[data["證券代碼"].str.contains(stock_id)]
    df = df.dropna()

    df["年月"] = df["年月"].astype(str)
    # df = df[df["年月"].str.contains(str(year))]

    # 只取年份
    df["year"] = df["年月"].str[:4]
    print(df)
    dfreg = df.loc[:, ["year", "收盤價(元)_年", "本益比", "股價淨值比"]]
    dfreg.fillna(value=-99999, inplace=True)  # 填補缺失值

    X = dfreg.loc[:, ["收盤價(元)_年", "本益比", "股價淨值比"]]
    X = X.set_index(df["year"])
    print(X)
    Y = np.where(df["收盤價(元)_年"].shift(-1) > df["收盤價(元)_年"], 1, -1)
    print(Y)

    # cv.fit(X, Y)
    # print(cv.best_params_["n_neighbors"])

    knn = KNeighborsClassifier(n_neighbors=int(2))
    X_train, X_test, Y_train, Y_test = SplitData(X, Y, year).split_data()
    # 訓練模型
    knn.fit(X_train, Y_train)
    # 預測
    pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, pred)
    print("Accuracy:", accuracy)
    accuracy_list.append(accuracy)

    # 交易信號
    df["Predict_Signal"] = knn.predict(X)
    # AAPL Cumulative Returns
    df["收盤價(元)_年"] = df["收盤價(元)_年"].astype(float)
    df["SPY_data_returns"] = np.log(df["收盤價(元)_年"] / df["收盤價(元)_年"].shift(1))  # 計算收益率
    Cumulative_SPY_data_returns = df[["SPY_data_returns"]].cumsum() * 100  # 計算累積收益率

    df["Startegy_returns"] = df["SPY_data_returns"] * df["Predict_Signal"].shift(
        1
    )  # 計算策略收益率
    Cumulative_Strategy_returns = df[["Startegy_returns"]].cumsum() * 100  # 計算累積策略收益率

    # Plot the results to visualize the performance
    get_ipython().run_line_magic("matplotlib", "inline")
    Cumulative_SPY_data_returns.plot()
    Cumulative_Strategy_returns.plot()
    plt.legend()
    plt.show()

print(accuracy_list)
