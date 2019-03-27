from numpy import zeros, array, mean
import numpy as np
from matplotlib.pyplot import plot, show, xlabel, ylabel,scatter

n = 56
b = array([0,0])

def estimate_coef(x, y):

    #mean of x and y vector
    m_x = mean(x)
    m_y = mean(y)

    #cross-deviation and deviation about x
    SS_xy = sum(y*x) - n*m_y*m_x
    SS_xx = sum(x*x) - n*m_x*m_x

    #regression coefficients
    b[1] = SS_xy / SS_xx
    b[0] = m_y - b[1]*m_x

    return(b[0], b[1])

#observations
x = np.array([3.08, 2.88, 2.78, 2.74, 2.85, 2.60, 2.41, 2.30, 2.31, 1.99, 1.82, 1.73, 1.70, 1.46, 1.38, 1.47, 1.76, 1.66, 1.72,
                  1.89, 2.08, 1.84, 1.81, 1.95, 2.13, 1.88, 1.86, 1.98, 2.14, 1.96, 1.98, 2.14, 2.31, 2.11, 2.08, 2.16, 2.18, 1.91,
                  1.82, 1.83, 1.86, 1.62, 1.54, 1.56, 1.60, 1.39, 1.32, 1.34, 1.32, 1.15, 1.13, 1.07, 1.12, 1.00, 0.96, 0.95])
y = array([56405,53648,55942,56086,58426,57231,56402,56984,73694,70154,73158,75642,65473,82351,80654,81004,29313,28720,
               70526,74256,56488,54867,50274,58463,67610,69213,63622,74148,77577,71236,57697,66701,75906,72085,75816,77176,
               52522,77795,69525,82517,92108,86412,80826,96991,105236,106486,92850,113486,127098,122628,109003,127606,139885,
               133160,130298,128546])

b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(b[0], b[1]))

#predicted response vector
y_pred = b[0] + b[1]*x

#plotting the actual points as scatter plot
scatter(x, y, color = "m", marker = "o")

#plotting the regression line
plot(x, y_pred, color = "g")

xlabel('wielkość bezrobocia [mln]')
ylabel('sprzedaż samochodów')
show()
