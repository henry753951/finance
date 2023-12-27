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

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dl = DataLoader()

top200_training = pd.read_csv("./data/top200_training.csv")

for row in top200_training.iterrows():
    dl.load_stocks(cache=True)

    df = dl.get_stocks_df(stock_id=dl.get_stocksID_ByName(row["簡稱"]), year=2019)

    # stock_id  Trading_Volume  Trading_money   open    max    min  close  spread  Trading_turnover    year    PER   PBR  equity    mkPrice       PSR    ROE    ROA  DailyReturn

    dfreg = df.loc[:, ["close", "Trading_Volume"]]  # Adjusted Close Price
    dfreg["HL_PCT"] = (
        (df["max"] - df["min"]) / df["close"] * 100.0
    )  # High Low Percentage
    dfreg["PCT_change"] = (df["close"] - df["open"]) / df["open"] * 100.0
    # Percentage Change

    dfreg.fillna(value=-99999, inplace=True)  # Drop missing value
    X = dfreg.loc[:, ["close", "HL_PCT", "PCT_change"]]  # 預測變量

    # 以年化報酬率為目標變量
    Y = np.where(dfreg["ReturnMean_year"] > dfreg["AllStockReturn"], 1, -1)  # 目標變量


# 分割訓練集和測試集 70%訓練集 30%測試集 寫成class
class SplitData:
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, split=0.7):
        self.X = X
        self.Y = Y
        self.split = split

    def split_data(self):
        split = int(len(self.X) * self.split)
        X_train = self.X[:split]
        Y_train = self.Y[:split]
        X_test = self.X[split:]
        Y_test = self.Y[split:]
        return X_train, Y_train, X_test, Y_test


knn = KNeighborsClassifier(n_neighbors=5)

# 訓練模型
X_train, Y_train, X_test, Y_test = SplitData(X, Y).split_data()
knn.fit(X_train, Y_train)

# 預測
pred = knn.predict(X_test)
print("預測結果:", pred)
print("實際結果:", Y_test)

# 準確率
accuracy = accuracy_score(Y_test, pred)
print("準確率:", accuracy)

# 繪圖
plt.plot(pred, label="pred")
plt.plot(Y_test, label="Y_test")

plt.legend()
plt.show()
