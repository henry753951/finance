import pprint

from sklearn.svm import SVC
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
import os

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
df = dl.getAllStocksPreYear(cols=["equity", "ROE", "ROA"], cols_deal=[None, None, None])
not_exist = []


stock_id = input()
accuracy_list = []

all = pd.read_csv("../data/top200_training.csv")
all_indexed = all.copy()
all_indexed.loc[:, "年月"] = all_indexed["年月"].astype(str).str.strip().str[:4].astype(int)
all_indexed.loc[:, "證券代碼"] = all_indexed["證券代碼"].astype(str)

all_indexed.set_index(["證券代碼", "年月"], inplace=True)


df = pd.merge(
    all_indexed,
    df,
    left_on=["證券代碼", "年月"],
    right_on=["stock_id", "year"],
    how="inner",
)
print(df)

# # 特徵選取
# rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)

# # 最佳k值

# # 訓練
# param_grid = {
#     "C": [1, 10, 100, 1000],
#     "gamma": [1, 0.1, 0.001, 0.0001],
#     "kernel": ["linear", "rbf"],
# }
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
