import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

excelfile = pd.read_excel('dane-zestawienie.xlsx')
columns = excelfile.iloc[:, [6, 1]]
print(columns)
dataset = np.array(columns)
X = dataset[:, 0:1]
print(X)
Y = dataset[:, 1]
print(Y)

regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, Y)

Y_Pred = regressor.predict([[100]])

X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))
plt.scatter(X, Y, color = 'r')
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'g')
plt.title('Random Forest Regression (sprzedaż samochodów - ludność)')
plt.xlabel('ludność')
plt.ylabel('sprzedaż samochodów')
plt.show()
