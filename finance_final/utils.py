import pandas as pd


def calcTurnover(df: pd.DataFrame):
    TotalTradingMoney = df["Trading_money"].sum()
    YearClosePrice = df["close"].iloc[0]
    Equity = df["equity"].iloc[0]
    turnover = TotalTradingMoney / (YearClosePrice * Equity)
    return turnover
