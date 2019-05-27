from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

excelfile = 'dane-zestawienie.xlsx'
data = pd.read_excel(excelfile)
x = data.iloc[:, 2:].values
print(x)
y = data.iloc[:, 1].values
print(y)

r_sq = 0

loo = LeaveOneOut()
loo.get_n_splits(x)

for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor(n_estimators = 57, random_state = 0)
    model.fit(X = x_train, y = y_train)
    y_pred = model.predict(x_test)

    r_sq = r_sq + model.score(x_train, y_train)

print('r^2:', r_sq/56)
