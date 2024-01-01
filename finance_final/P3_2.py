import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics


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
from sklearn.metrics import accuracy_score


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time


# 拆分資料
class SplitData:
    def __init__(self, X: pd.DataFrame, Y: np.ndarray, year: int):
        self.X = X
        self.Y = Y
        self.year = year

    def split_data(self):
        X_train = self.X[self.X.index <= self.year]
        X_test = self.X[self.X.index > self.year]
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


stocks = list(set(all["證券代碼"].to_list()))  # 取出所有股票代碼


for year in range(1997, 2010):
    X = pd.DataFrame()
    Y = np.array([])
    print(year)
    for stock in stocks:
        df = all[all["證券代碼"] == stock]
        # 只保留df["年月"]的年份
        df.loc[:, "年月"] = df["年月"].astype(str).str.strip().str[:4].astype(int)
        df.set_index("年月", inplace=True)
        df.sort_index(inplace=True)

        X = pd.concat([X, df])  # 連接資料
        X.drop("簡稱", axis=1, inplace=True)
        Y_values = df["ReturnMean_year_Label"].shift(-1)
        Y_values.dropna(inplace=True)
        X = X[:-1]
        Y = np.concatenate([Y, Y_values.values])
        # 將數據分割為訓練集和測試集

    X_train, X_test, y_train, y_test = SplitData(X, Y, year).split_data()
    rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    rf.fit(X_train, y_train)
    print("特徵重要程度: ", rf.feature_importances_)
    feature_index = np.array(rf.feature_importances_.argsort()[-10:][::-1])
    feature_index.sort()
    new_feature = [X.columns[i] for i in feature_index]
    print("feature_index", feature_index)
    X_train = X_train.loc[:][new_feature]
    X_test = X_test.loc[:][new_feature]

    # X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    #     X_new, Y, test_size=0.2, random_state=42
    # )

    # 建立SVM模型
    svc_model = SVC()
    svc_model.fit(X_train, y_train)

    # 預測測試集
    y_pred = svc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


print(accuracy_list)
result = [
    {"cols": col_name[i], "accuracy": accuracy_list[i]}
    for i in range(len(accuracy_list))
]

# write to json

with open("P2result.json", "w", encoding="utf8") as outfile:
    json.dump(result, outfile, ensure_ascii=False, indent=4)
print(best_accuracy, best_col)
