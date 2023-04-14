import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate

#Flow rate = 20ml/min
df = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
# df = df[:-1]
df_CtbyC0 = df.iloc[:,1]
df_Time = df.iloc[:,0]

#Kinetic models
df_Thomas = np.log((1/df_CtbyC0)-1)
df_AdamsBohart = np.log(df_CtbyC0)
df_YoonNelson = np.log(1/((1/df_CtbyC0)-1))   

box = {'facecolor': 'none',
       'edgecolor': 'black',
       'boxstyle': 'round'
      }





#Linear regression for Thomas model
x_Th = np.array(df_Time).reshape(-1,1)
y_Th = np.array(df_Thomas)

model_Th = LinearRegression()
model_Th.fit(x_Th, y_Th)
model_Th.intercept_  
model_Th.coef_ 

#Parameters for Thomas
r_sq_Th = model_Th.score(x_Th,y_Th)
y_pred_Th = model_Th.predict(x_Th)

k_Th = -model_Th.coef_ / 20
q_0 = model_Th.intercept_ / ((k_Th * 2.501) / 20) 

mse_Th = mean_squared_error(df_Thomas, y_pred_Th)

df_Th = pd.DataFrame({'Actual': y_Th, 'Predicted': y_pred_Th})
rss_Th = np.sum(np.square(df_Th['Predicted'] - df_Th['Actual']))

#Plotting
plt.scatter(x_Th, y_Th, color = "m", marker = "o", s = 30)
plt.plot(x_Th, y_pred_Th, color = "r")
plt.text(1, -8, 'Rate constant = %s \nAdsorption capactiy = %s \nCoefficient of determination = %s \
         ' % (k_Th, q_0, r_sq_Th), bbox=box)





#Linear regression for Adams-Bohart model
x_AB = np.array(df_Time).reshape(-1,1)
y_AB = np.array(df_AdamsBohart)

model_AB = LinearRegression()
model_AB.fit(x_AB, y_AB)
model_AB.intercept_  
model_AB.coef_ 

#Parameters for Adams-Bohart
r_sq_AB = model_AB.score(x_AB,y_AB)
y_pred_AB = model_AB.predict(x_AB)

k_AB = model_AB.coef_ / 20
N_0 = model_AB.intercept_ /(-k_AB * (5 / 0.392))

mse_AB = mean_squared_error(df_AdamsBohart, y_pred_AB)

df_AB = pd.DataFrame({'Actual': y_AB, 'Predicted': y_pred_AB})
rss_AB = np.sum(np.square(df_AB['Predicted'] - df_AB['Actual']))

#Plotting
plt.scatter(x_AB, y_AB, color = "m", marker = "o", s = 30)
plt.plot(x_AB, y_pred_AB, color = "r")
plt.text(0, 0.5, 'Rate constant = %s \nSaturation concentration = %s \nCoefficient of determination = %s \
         ' % (k_AB, N_0, r_sq_AB), bbox=box)





#Linear regression for Yoon-Nelson model
x_YN = np.array(df_Time).reshape(-1,1) 
y_YN = np.array(df_YoonNelson)

model_YN = LinearRegression()
model_YN.fit(x_YN, y_YN)
model_YN.intercept_  
model_YN.coef_  

#Parameters for Yoon-Nelson 
r_sq_YN = model_YN.score(x_YN,y_YN)
y_pred_YN = model_YN.predict(x_YN)

k_YN = model_YN.coef_ 
torque = model_YN.intercept_ / (-1 * k_YN)

mse_YN = mean_squared_error(df_YoonNelson, y_pred_YN)

df_YN = pd.DataFrame({'Actual': y_YN, 'Predicted': y_pred_YN})
rss_YN = np.sum(np.square(df_YN['Predicted'] - df_YN['Actual']))

#Plotting
plt.scatter(x_YN, y_YN, color = "m", marker = "o", s = 30)
plt.plot(x_YN, y_pred_YN, color = "r")
plt.text(0, 8, 'Rate constant = %s \nTime required for 50 percent adsorbate breakthrough = %s \
         \nCoefficient of determination = %s ' % (k_YN, torque, r_sq_YN), bbox=box)





plt.xlabel('Time')
plt.ylabel('Ct/C0')
# plt.show()

data = [['Thomas', 20, 20, 5, k_Th, q_0, r_sq_Th, mse_Th, rss_Th], \
       ['Adams-Bohart', 20, 20, 5, k_AB, N_0, r_sq_AB, mse_AB, rss_AB], \
       ['Yoon-Nelson', 20, 20, 5, k_YN, torque, r_sq_YN, mse_YN, rss_YN]]

headers=["Model", "C0 (mg/L)", "Q (mL/min)", "Z (cm)", \
             "Rate constant", "Parameter", "R_squared", "MSE", "RSS"]
print (tabulate(data, headers))