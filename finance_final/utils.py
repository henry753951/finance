import pandas as pd
import numpy as np


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


def calcTurnover(df: pd.DataFrame):
    TotalTradingMoney = df["Trading_money"].sum()
    YearClosePrice = df["close"].iloc[0]
    Equity = df["equity"].iloc[0]
    turnover = TotalTradingMoney / (YearClosePrice * Equity)
    return turnover


def normalization(data):
    data = data.copy()
    for col in data.columns[2:-2]:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    return data


def getStockByYear(year, stock_id, all_indexed) -> dict or bool:
    try:
        find: pd.DataFrame = all_indexed.loc[(stock_id, year)]
    except KeyError:
        return False
    return find.to_dict()


def strategy(Y_pred: np.ndarray, X_test_copy: pd.DataFrame, all_indexed):
    # X_test_strategy = 2000 - 2005
    # Y_pred = 2001 - 2006

    # for loop skip first year
    # last year unknown
    output = []
    for index, row in enumerate(X_test_copy.tail(len(Y_pred)).iterrows()):
        if Y_pred[index] == 1:
            NextYearStockData = getStockByYear(row[0] + 1, row[1]["證券代碼"], all_indexed)
            if not NextYearStockData:
                continue
            output.append(
                {
                    "year": row[0] + 1,
                    "stock": row[1]["證券代碼"],
                    "stock_name": row[1]["簡稱"],
                    "open_price": float(row[1]["收盤價(元)_年"]),
                    "close_price": float(NextYearStockData["收盤價(元)_年"]),
                    "return": float(NextYearStockData["收盤價(元)_年"]) / float(row[1]["收盤價(元)_年"]),
                }
            )
    print("output: ", output)
    return output
