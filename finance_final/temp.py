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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
        index = self.X[self.X.index.year == self.year].index[0]
        row_number = self.X.index.get_loc(index)
        X_train = self.X[self.X.index.year < self.year]
        X_test = self.X[self.X.index.year >= self.year]
        Y_train = self.Y[:row_number]
        Y_test = self.Y[row_number:]
        return X_train, X_test, Y_train, Y_test


dl = DataLoader()
dl.load_stocks(cache=True)
top200_training = pd.read_csv("../data/top200_training.csv")
not_exist = []


stock_id = input()
accuracy_list = []
years = dl.get_stock_Years_byID(stock_id)
for year in years[1:]:
    print(year)
    df = dl.get_stocks_df(stock_id=stock_id)
    dfreg = df.loc[:, ["close", "Trading_Volume"]]  # Adjusted Close Price
    dfreg["HL_PCT"] = (df["max"] - df["min"]) / df["close"] * 100.0
    # High Low Percentage

    dfreg["PCT_change"] = (df["close"] - df["open"]) / df["open"] * 100.0
    # Percentage Change

    dfreg.fillna(value=-99999, inplace=True)  # Drop missing value
    X = dfreg.loc[:, ["close", "HL_PCT", "PCT_change"]]  # 預測變量
    print(X)
    # 以為目標變量
    Y = np.where(df["close"].shift(-1) > df["close"], 1, -1)
    cv.fit(X, Y)
    # print("k=", cv.best_params_["n_neighbors"])
    knn = KNeighborsClassifier(n_neighbors=cv.best_params_["n_neighbors"])

    # 訓練模型
    X_train, X_test, Y_train, Y_test = SplitData(X, Y, year).split_data()
    print(X_train)
    knn.fit(X_train, Y_train)

    # 預測
    pred = knn.predict(X_test)
    # print("預測結果:", pred)
    # print("實際結果:", Y_test)

    # 準確率
    accuracy = accuracy_score(Y_test, pred)
    print("準確率:", accuracy)

    accuracy_list.append((year, accuracy))

    # 交易信號
    df["Predict_Signal"] = knn.predict(X)
    # AAPL Cumulative Returns
    df["SPY_data_returns"] = np.log(df["close"] / df["close"].shift(1))
    Cumulative_SPY_data_returns = df[["SPY_data_returns"]].cumsum() * 100

    df["Startegy_returns"] = df["SPY_data_returns"] * df["Predict_Signal"].shift(1)
    Cumulative_Strategy_returns = df[["Startegy_returns"]].cumsum() * 100
    # Plot the results to visualise the performance
    plt.figure(figsize=(10, 5))
    plt.plot(Cumulative_SPY_data_returns, color="r", label="SPY_data Returns")
    plt.plot(Cumulative_Strategy_returns, color="g", label="Strategy Returns")
    plt.legend()
    plt.show()
# 繪圖
accuracy_list = np.array(accuracy_list)
plt.plot(accuracy_list[:, 0], accuracy_list[:, 1])
plt.title(dl.get_stocksID_ByName(str(row[1]["簡稱"]).strip()))
plt.xlabel("Year")
plt.ylabel("Accuracy")
plt.show()
