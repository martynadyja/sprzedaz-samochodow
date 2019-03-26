from numpy import zeros, array, mean
import numpy as np
from matplotlib.pyplot import plot, show, xlabel, ylabel,scatter

n=56
b=array([0,0])

def estimate_coef(x, y):

    # mean of x and y vector
    m_x = mean(x)
    m_y = mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = sum(y*x) - n*m_y*m_x
    SS_xx = sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b[1] = SS_xy / SS_xx
    b[0] = m_y - b[1]*m_x

    return(b[0], b[1])

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    scatter(x, y, color = "m",
               marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plot(x, y_pred, color = "g")

    # putting labels
    xlabel('średni koszt kredytu')
    ylabel('sprzedaż samochodów')

    # function to show plot
    show()

def main():
    # observations
    x = np.array([0.2342, 0.2279, 0.2262, 0.2201, 0.1963, 0.1879, 0.1908, 0.1965, 0.2158, 0.2112, 0.2144, 0.2121, 0.2137, 0.2212,
                  0.2285, 0.2306, 0.2291, 0.2287, 0.2281, 0.2197, 0.2182, 0.2195, 0.2174, 0.2108, 0.2205, 0.2230, 0.2166, 0.2129,
                  0.2210, 0.2155, 0.2273, 0.2279, 0.2181, 0.2105, 0.2062, 0.1993, 0.1991, 0.2013, 0.1970, 0.1733, 0.1623, 0.1545,
                  0.1559, 0.1520, 0.1516, 0.1487, 0.1556, 0.1529, 0.1452, 0.1452, 0.1468, 0.1377, 0.1375, 0.1391, 0.1384, 0.1351])
    y = array([56405,53648,55942,56086,58426,57231,56402,56984,73694,70154,73158,75642,65473,82351,80654,81004,29313,28720,
               70526,74256,56488,54867,50274,58463,67610,69213,63622,74148,77577,71236,57697,66701,75906,72085,75816,77176,
               52522,77795,69525,82517,92108,86412,80826,96991,105236,106486,92850,113486,127098,122628,109003,127606,139885,
               133160,130298,128546])

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()

