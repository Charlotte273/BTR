import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

#PART 1-----
#Flow rate = 20ml/min
df_20 = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
df_20 = df_20[:-1]
df_20_CtbyC0 = df_20.iloc[:,1]
df_20_Time = df_20.iloc[:,0]

#Kinetic models for Q = 20ml/min
df_Thomas_20 = np.log((1/df_20_CtbyC0)-1)
df_YoonNelson_20 = np.log(1/((1/df_20_CtbyC0)-1))

#Linear regression for Yoon-Nelson model
x_YN20 = np.array(df_20_Time).reshape(-1,1) #independent value
y_YN20 = np.array(df_YoonNelson_20) #dependent value

model_YN20 = LinearRegression()
model_YN20.fit(x_YN20, y_YN20)
model_YN20.intercept_ #b0 coefficient 
model_YN20.coef_ #b1 coefficient 

#determine how good the linear model fits the data 
#the value close to 1 or exaclty 1 = the linear model is a good fit for data
r_sq_YN20 = model_YN20.score(x_YN20,y_YN20)

#predict the observations 
# f(x) = model.intercept_ + model.coef_ * x
y_YN20_pred = model_YN20.predict(x_YN20)

#Linear regression for Thomas model
x_T20 = np.array(df_20_Time).reshape(-1,1)
y_T20 = np.array(df_Thomas_20)

model_T20 = LinearRegression()
model_T20.fit(x_T20, y_T20)
model_T20.intercept_  
model_T20.coef_ 

r_sq_T20 = model_T20.score(x_T20,y_T20)
y_pred_T20 = model_T20.predict(x_T20)

# #PLOTTING
# #plotting the actual points
# plt.scatter(x_T20, y_T20, color = "m", marker = "o", s = 30)
# plt.scatter(x_YN20, y_YN20, color = "m", marker = "o", s = 30)

# #plotting the regression line
# plt.plot(x_YN20, y_YN20_pred, color = "g")
# plt.plot(x_T20, y_pred_T20, color = "r")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

torque = model_T20.intercept_ / (-1 * model_T20.coef_)
print(torque)