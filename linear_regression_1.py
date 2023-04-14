import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Flow rate = 20ml/min
df_20 = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
df_20 = df_20[:-1]
df_20_CtbyC0 = df_20.iloc[:,1]
df_20_Time = df_20.iloc[:,0]

#Kinetic models for Q = 20ml/min
df_Thomas_20 = np.log((1/df_20_CtbyC0)-1)
df_YoonNelson_20 = np.log(1/((1/df_20_CtbyC0)-1))

#Linear regression for Yoon-Nelson
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(df_20_CtbyC0)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)
  
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
  
def main():
    # observations / data
    x = df_20_Time
    y = df_YoonNelson_20
  
    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
  
    # plotting regression line
    plot_regression_line(x, y, b)
  
if __name__ == "__main__":
    main()