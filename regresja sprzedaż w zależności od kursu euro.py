from numpy import zeros, array, mean
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
    scatter(x, y, color = "m", marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plot(x, y_pred, color = "g")

    # putting labels
    xlabel('kurs euro')
    ylabel('sprzedaż samochodów')

    # function to show plot
    show()

def main():
    # observations
    x = array([4.02,4.13,4.02,3.91,3.83,3.94,3.95,3.84,3.88,3.80,3.78,3.65,3.57,3.41,3.31,3.77,4.49,4.44,4.19,4.17,3.98,
               4.01,4.01,3.96,3.94,3.95,4.15,4.42,4.22,4.26,4.13,4.11,4.15,4.19,4.24,4.18,4.18,4.16,4.17,4.21,4.19,4.08,
               4.18,4.26,4.36,4.37,4.34,4.37,4.32,4.21,4.25,4.23,4.18,4.26,4.30,4.29])
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





