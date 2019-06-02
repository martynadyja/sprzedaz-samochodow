import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

#wszystko
excel_file = 'dane.xlsx'
data = pd.read_excel(excel_file)
x = data.iloc[:, :-1].values
y = data.iloc[:, 6].values
#liczba ludności
x1 = data.iloc[:,4:-2].values
#ceny benzyny
x2 = data.iloc[:,2:-4].values
#średnie wynagrodzenie
x3 = data.iloc[:, :-6].values
#kurs euro
x4 = data.iloc[:,3:-3].values
#wielkość bezrobocia
x5 = data.iloc[:,1:-5].values
#koszt kredytu
x6 = data.iloc[:,5:-1].values

print('modele:')
print('1 - regresja gaussa')
print('2 - regresja liniowa pierwszego stopnia')
print('3 - regresja wielomianowa drugiego stopnia')
print('4 - regresja wielomianowa trzeciego stopnia')
print('5 - random forest regression')
print('zmienne:')
print('1 - liczba ludności')
print('2 - ceny beznyny')
print('3 - średnie wynagrodzenie')
print('4 - kurs euro')
print('5 - wielkość bezrobocia')
print('6 - koszt kredytu')
print('7 - wszystko')

print('ile modeli porówanać:')
a=int(input())
tab=np.zeros(a,int)
zmienne=np.zeros(a,int)

for i in range(a):
    print('proszę wybrać model: ')
    temp=int(input())
    tab[i]=temp
    if tab[i]>5:
        print('zły numer, proszę podać jeszcze raz')
        temp=int(input())
        tab[i]=temp
    print('jakie zmienne')
    temp=int(input())
    zmienne[i]=temp
    if zmienne[i]>7:
        print('zły numer, proszę podać jeszcze raz')
        temp=int(input())
        zmienne[i]=temp

nazwa_modelu = []
wartość_r2 = []

for i in range(a):
    if tab[i]==1:
        if zmienne[i]==1:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x1):
                x1_train, x1_test = x1[train_index], x1[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x1_train, y_train)

                r_sq = r_sq + gpr.score(x1_train, y_train)

                nazwa = 'model gaussa, liczba ludności'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, liczba ludności:')
            print('r^2:', r_sq/56)
        if zmienne[i]==2:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x2)

            for train_index, test_index in loo.split(x2):
                x2_train, x2_test = x2[train_index], x2[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x2_train, y_train)

                r_sq = r_sq + gpr.score(x2_train, y_train)

                nazwa = 'model gaussa, ceny benzyny'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, ceny benzyny:')
            print('r^2:', r_sq/56)
        if zmienne[i]==3:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x3)

            for train_index, test_index in loo.split(x3):
                x3_train, x3_test = x3[train_index], x3[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x3_train, y_train)

                r_sq = r_sq + gpr.score(x3_train, y_train)

                nazwa = 'model gaussa, średnie wynagrodzenie'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, średnie wynagrodzenie:')
            print('r^2:', r_sq/56)
        if zmienne[i]==4:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x4)

            for train_index, test_index in loo.split(x4):
                x4_train, x4_test = x4[train_index], x4[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x4_train, y_train)

                r_sq = r_sq + gpr.score(x4_train, y_train)

                nazwa = 'model gaussa, kurs euro'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, kurs euro:')
            print('r^2:', r_sq/56)
        if zmienne[i]==5:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x5)

            for train_index, test_index in loo.split(x5):
                x5_train, x5_test = x5[train_index], x5[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x5_train, y_train)

                r_sq = r_sq + gpr.score(x5_train, y_train)

                nazwa = 'model gaussa, wielkość bezrobocia'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, wielkość bezrobocia:')
            print('r^2:', r_sq/56)
        if zmienne[i]==6:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x6)

            for train_index, test_index in loo.split(x6):
                x6_train, x6_test = x6[train_index], x6[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x6_train, y_train)

                r_sq = r_sq + gpr.score(x6_train, y_train)

                nazwa = 'model gaussa, koszt kredytu'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, koszt kredytu:')
            print('r^2:', r_sq/56)
        if zmienne[i]==7:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x)

            for train_index, test_index in loo.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kernel = DotProduct() + WhiteKernel()
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1, random_state=0).fit(x_train, y_train)

                r_sq = r_sq + gpr.score(x_train, y_train)

                nazwa = 'model gaussa, wszystkie zmienne'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model gaussa, wszystkie zmienne:')
            print('r^2:', r_sq/56)
    if tab[i]==2:
        if zmienne[i]==1:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x1):
                x1_train, x1_test = x1[train_index], x1[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x1_train, y_train)

                r_sq = r_sq + model.score(x1_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, liczba ludności'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji liniowej pierwszego stopnia, liczba ludności:')
            print('r^2:', r_sq/56)
        if zmienne[i]==2:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x2)

            for train_index, test_index in loo.split(x2):
                x2_train, x2_test = x2[train_index], x2[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x2_train, y_train)

                r_sq = r_sq + model.score(x2_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, ceny benzyny'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji liniowej pierwszego stopnia, ceny benzyny:')
            print('r^2:', r_sq/56)
        if zmienne[i]==3:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x3)

            for train_index, test_index in loo.split(x3):
                x3_train, x3_test = x3[train_index], x3[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x3_train, y_train)

                r_sq = r_sq + model.score(x3_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, średnie wynagrodzenie'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji liniowej pierwszego stopnia, średnie wynagrodzenie:')
            print('r^2:', r_sq/56)
        if zmienne[i]==4:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x4)

            for train_index, test_index in loo.split(x4):
                x4_train, x4_test = x4[train_index], x4[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x4_train, y_train)

                r_sq = r_sq + model.score(x4_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, kurs euro'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji liniowej pierwszego stopnia, kurs euro:')
            print('r^2:', r_sq/56)
        if zmienne[i]==5:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x5)

            for train_index, test_index in loo.split(x5):
                x5_train, x5_test = x5[train_index], x5[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x5_train, y_train)

                r_sq = r_sq + model.score(x5_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, wielkość bezrobocia'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji liniowej pierwszego stopnia, wielkość bezrobocia:')
            print('r^2:', r_sq/56)
        if zmienne[i]==6:

            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x6)

            for train_index, test_index in loo.split(x6):
                x6_train, x6_test = x6[train_index], x6[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x6_train, y_train)

                r_sq = r_sq + model.score(x6_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, koszt kredytu'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji liniowej pierwszego stopnia, koszt kredytu:')
            print('r^2:', r_sq/56)
        if zmienne[i]==7:
            r_sq=0

            loo = LeaveOneOut()
            loo.get_n_splits(x)

            for train_index, test_index in loo.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x_train, y_train)

                r_sq = r_sq + model.score(x_train, y_train)

                nazwa = 'model regresji liniowej pierwszego stopnia, wszystkie zmienne'

                y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model regresji liniowej pierwszego stopnia, wszystkie zmienne:')
            print('r^2:', r_sq/56)
    if tab[i]==3:
        if zmienne[i]==1:

            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x1)
            x1_ = transformer.transform(x1)

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x1_):
                x1_train, x1_test = x1_[train_index], x1_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x1_train, y_train)

                r_sq = r_sq + model.score(x1_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, liczba ludności'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej drugiego stopnia, liczba ludności:')
            print('r^2:', r_sq/56)
        if zmienne[i]==2:

            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x2)
            x2_ = transformer.transform(x2)

            loo = LeaveOneOut()
            loo.get_n_splits(x2)

            for train_index, test_index in loo.split(x2_):
                x2_train, x2_test = x2_[train_index], x2_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x2_train, y_train)

                r_sq = r_sq + model.score(x2_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, ceny benzyny'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej drugiego stopnia, ceny benzyny:')
            print('r^2:', r_sq/56)
        if zmienne[i]==3:

            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x3)
            x3_ = transformer.transform(x3)

            loo = LeaveOneOut()
            loo.get_n_splits(x3)

            for train_index, test_index in loo.split(x3_):
                x3_train, x3_test = x3_[train_index], x3_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x3_train, y_train)

                r_sq = r_sq + model.score(x3_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, średnie wynagrodzenie'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej drugiego stopnia, średnie wynagrodzenie:')
            print('r^2:', r_sq/56)
        if zmienne[i]==4:

            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x4)
            x4_ = transformer.transform(x4)

            loo = LeaveOneOut()
            loo.get_n_splits(x4)

            for train_index, test_index in loo.split(x4_):
                x4_train, x4_test = x4_[train_index], x4_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x4_train, y_train)

                r_sq = r_sq + model.score(x4_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, kurs euro'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej drugiego stopnia, kurs euro:')
            print('r^2:', r_sq/56)
        if zmienne[i]==5:

            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x5)
            x5_ = transformer.transform(x5)

            loo = LeaveOneOut()
            loo.get_n_splits(x5)

            for train_index, test_index in loo.split(x5_):
                x5_train, x5_test = x5_[train_index], x5_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x5_train, y_train)

                r_sq = r_sq + model.score(x5_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, wielkość bezrobocia'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej drugiego stopnia, wielkość bezrobocia:')
            print('r^2:', r_sq/56)
        if zmienne[i]==6:

            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x6)
            x6_ = transformer.transform(x6)

            loo = LeaveOneOut()
            loo.get_n_splits(x6)

            for train_index, test_index in loo.split(x6_):
                x6_train, x_test = x6_[train_index], x6_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x6_train, y_train)

                r_sq = r_sq + model.score(x6_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, koszt kredytu'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej drugiego stopnia, koszt kredytu:')
            print('r^2:', r_sq/56)
        if zmienne[i]==7:
            r_sq=0

            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x)
            x_ = transformer.transform(x)

            loo = LeaveOneOut()
            loo.get_n_splits(x)

            for train_index, test_index in loo.split(x_):
                x_train, x_test = x_[train_index], x_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x_train, y_train)

                r_sq = r_sq + model.score(x_train, y_train)

                nazwa = 'model regresji wielomianowej drugiego stopnia, wszystkie zmienne'

                y_pred = model.predict(x_test)
                    #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model regresji wielomianowej drugiego stopnia, wszystkie zmienne:')
            print('r^2:', r_sq/56)
    if tab[i]==4:
        if zmienne[i]==1:

            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x1)
            x1_ = transformer.transform(x1)

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x1_):
                x1_train, x1_test = x1_[train_index], x1_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x1_train, y_train)

                r_sq = r_sq + model.score(x1_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, liczba ludności'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej trzeciego stopnia, liczba ludności:')
            print('r^2:', r_sq/56)
        if zmienne[i]==2:

            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x2)
            x2_ = transformer.transform(x2)

            loo = LeaveOneOut()
            loo.get_n_splits(x2)

            for train_index, test_index in loo.split(x2_):
                x2_train, x2_test = x2_[train_index], x2_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x2_train, y_train)

                r_sq = r_sq + model.score(x2_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, ceny benzyny'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej trzeciego stopnia, ceny benzyny:')
            print('r^2:', r_sq/56)
        if zmienne[i]==3:

            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x3)
            x3_ = transformer.transform(x3)

            loo = LeaveOneOut()
            loo.get_n_splits(x3)

            for train_index, test_index in loo.split(x3_):
                x3_train, x3_test = x3_[train_index], x3_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x3_train, y_train)

                r_sq = r_sq + model.score(x3_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, średnie wynagrodzenie'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej trzeciego stopnia, średnie wynagrodzenie:')
            print('r^2:', r_sq/56)
        if zmienne[i]==4:

            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x4)
            x4_ = transformer.transform(x4)

            loo = LeaveOneOut()
            loo.get_n_splits(x4)

            for train_index, test_index in loo.split(x4_):
                x4_train, x4_test = x4_[train_index], x4_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x4_train, y_train)

                r_sq = r_sq + model.score(x4_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, kurs euro'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej trzeciego stopnia, kurs euro:')
            print('r^2:', r_sq/56)
        if zmienne[i]==5:

            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x5)
            x5_ = transformer.transform(x5)

            loo = LeaveOneOut()
            loo.get_n_splits(x5)

            for train_index, test_index in loo.split(x5_):
                x5_train, x5_test = x5_[train_index], x5_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x5_train, y_train)

                r_sq = r_sq + model.score(x5_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, wielkość bezrobocia'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej trzeciego stopnia, wielkość bezrobocia:')
            print('r^2:', r_sq/56)
        if zmienne[i]==6:

            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x6)
            x6_ = transformer.transform(x6)

            loo = LeaveOneOut()
            loo.get_n_splits(x6)

            for train_index, test_index in loo.split(x6_):
                x6_train, x_test = x6_[train_index], x6_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x6_train, y_train)

                r_sq = r_sq + model.score(x6_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, koszt kredytu'

                #y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model regresji wielomianowej trzeciego stopnia, koszt kredytu:')
            print('r^2:', r_sq/56)
        if zmienne[i]==7:
            r_sq=0

            transformer = PolynomialFeatures(degree=3, include_bias=False)
            transformer.fit(x)
            x_ = transformer.transform(x)

            loo = LeaveOneOut()
            loo.get_n_splits(x_)

            for train_index, test_index in loo.split(x_):
                x_train, x_test = x_[train_index], x_[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LinearRegression().fit(x_train, y_train)

                r_sq = r_sq + model.score(x_train, y_train)

                nazwa = 'model regresji wielomianowej trzeciego stopnia, wszystkie zmienne'

                y_pred = model.predict(x_test)
                #print(test_index+1, ':',y_pred)
            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model regresji wielomianowej trzeciego stopnia, wszystkie zmienne:')
            print('r^2:', r_sq/56)
    if tab[i] == 5:
        if zmienne[i] == 1:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x1):
                x1_train, x1_test = x1[train_index], x1[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x1_train, y = y_train)
                #y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x1_train, y_train)

                nazwa = 'model random forest regression, liczba ludności'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model random forest regression, liczba ludności')
            print('r^2:', r_sq/56)
        if zmienne[i] == 2:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x2)

            for train_index, test_index in loo.split(x2):
                x2_train, x2_test = x2[train_index], x2[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x2_train, y = y_train)
                #y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x2_train, y_train)

                nazwa = 'model random forest regression, ceny benzyny'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model random forest regression, ceny benzyny')
            print('r^2:', r_sq/56)
        if zmienne[i] == 3:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x3)

            for train_index, test_index in loo.split(x3):
                x3_train, x3_test = x3[train_index], x3[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x3_train, y = y_train)
                #y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x3_train, y_train)

                nazwa = 'model random forest regression, średnie wynagrodzenie'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model random forest regression, średnie wynagrodzenie')
            print('r^2:', r_sq/56)
        if zmienne[i] == 4:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x4):
                x4_train, x4_test = x4[train_index], x4[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x4_train, y = y_train)
                #y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x4_train, y_train)

                nazwa = 'model random forest regression, kurs euro'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model random forest regression, kurs euro')
            print('r^2:', r_sq/56)
        if zmienne[i] == 5:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x1)

            for train_index, test_index in loo.split(x5):
                x5_train, x5_test = x5[train_index], x5[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x5_train, y = y_train)
                #y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x5_train, y_train)

                nazwa = 'model random forest regression, wielkość bezrobocia'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model random forest regression, wielkość bezrobocia')
            print('r^2:', r_sq/56)
        if zmienne[i] == 6:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x6)

            for train_index, test_index in loo.split(x6):
                x6_train, x6_test = x6[train_index], x6[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x6_train, y = y_train)
                #y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x6_train, y_train)

                nazwa = 'model random forest regression, koszt kredytu'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('model random forest regression, koszt kredytu')
            print('r^2:', r_sq/56)
        if zmienne[i] == 7:
            r_sq = 0

            loo = LeaveOneOut()
            loo.get_n_splits(x)

            for train_index, test_index in loo.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators = 100, random_state = 0)
                model.fit(X = x_train, y = y_train)
                y_pred = model.predict(x_test)

                r_sq = r_sq + model.score(x_train, y_train)

                nazwa = 'model random forest regression'

            wartość_r2.append(r_sq/56)
            nazwa_modelu.append(nazwa)
            print('Model random forest regression:')
            print('r^2:', r_sq/56)

data = {'model':nazwa_modelu, 'r^2': wartość_r2}
df = pd.DataFrame(data)
print(df)

tabela = "tabela_wyniki.csv"
df.to_csv(path_or_buf = tabela , na_rep = 'NaN',
             columns = None, header = True,
             index = False, mode = 'w',
             encoding = 'utf-8',
             line_terminator = '\n',)
