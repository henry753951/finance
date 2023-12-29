from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 將資料分割為訓練集及測試集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42) 

# 建立決策樹模型
decisionTreeModel = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)

# 使用訓練資料訓練模型
decisionTreeModel.fit(X_train, y_train)

# 使用訓練資料預測分類 
train_predicted = decisionTreeModel.predict(X_train)

# 使用測試資料預測分類
test_predicted = decisionTreeModel.predict(X_test)

# 計算訓練集的準確率
train_accuracy = accuracy_score(y_train, train_predicted)
print("Training Accuracy:", train_accuracy) 

# 計算測試集的準確率
test_accuracy = accuracy_score(y_test, test_predicted)
print("Testing Accuracy:", test_accuracy)