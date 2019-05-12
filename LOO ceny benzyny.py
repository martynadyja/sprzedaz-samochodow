from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

excelfile = pd.read_excel('dane-zestawienie.xlsx')
columns = excelfile.iloc[:, [3, 1]]
dataset = np.array(columns)
X = dataset[:, 0:1]
print(X)
y = dataset[:, 1]
print(y)

y_tests = []
y_preds = []

loo = LeaveOneOut()
loo.get_n_splits(X)

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor(n_estimators = 57, random_state = 0)
    model.fit(X = X_train, y = y_train)
    y_pred = model.predict(X_test)

    y_tests += list(y_test)
    y_preds += list(y_pred)

    X_Grid = np.arange(min(X), max(X), 0.01)
    X_Grid = X_Grid.reshape((len(X_Grid), 1))
    plt.scatter(X_test, y_test, color = 'r', label = 'x testowy')
    plt.scatter(X_train, y_train, color = 'k', label = 'wartości treningowe')
    plt.plot(X_Grid, model.predict(X_Grid), color = 'g')
    plt.title('Random Forest Regression (sprzedaż samochodów - ceny benzyny)')
    plt.xlabel('ceny benzyny')
    plt.ylabel('sprzedaż samochodów')
    plt.legend(loc = 'upper left')
    plt.show()

rr = metrics.r2_score(y_tests, y_preds)
ms_error = metrics.mean_squared_error(y_tests, y_preds)

print("Leave One Out Cross Validation")
print("R^2: {:.5f}%, MSE: {:.5f}".format(rr * 100, ms_error))

scores = cross_val_score(model, X, y, cv = 56, scoring= 'r2')
print("Cross-validated scores:", scores)
print("Average: ", scores.mean())
print("Variance: ", np.std(scores))



