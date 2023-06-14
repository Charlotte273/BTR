import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate

#PART 1: PLOT EXPERIMENTAL BREAKTHROUGH CURVES

#Time (min)
time = np.array([10, 20, 30, 60, 120, 180, 240])

#Flow rate 
Q = 60 #ml/min

#Mass of the adsorbent - activated carbon
m = 200 #g

#Effluent concentration (mmol/L) - the first value is influent concentration C0
observed_bicarbonate = np.array([30.9, 31.0, 31.4, 30.9, 31.0, 30.8, 31.0, 30.4])
observed_magnesium = np.array([0.70, 0.33, 0.40, 0.42, 0.46, 0.50, 0.54, 0.55])
observed_sodium = np.array([137, 139, 139, 138, 138, 138, 138, 138])
observed_calcium = np.array([1.34, 0.86, 1.04, 1.14, 1.22, 1.32, 1.35, 1.30])
observed_glucose = np.array([106.9, 5.7, 37.4, 68.3, 98.7, 106.1, 107.2, 106.2])

#Experimental Ct/C0 
exp_bicarbonate = np.array(observed_bicarbonate[1:] / observed_bicarbonate[0])
exp_magnesium = np.array(observed_magnesium[1:] / observed_magnesium[0])
exp_sodium = np.array(observed_sodium[1:] / observed_sodium[0])
exp_calcium = np.array(observed_calcium[1:] / observed_calcium[0])
exp_glucose = np.array(observed_glucose[1:] / observed_glucose[0])

# plt.scatter(time, exp_bicarbonate, label='Experimental bicarbonate')
# plt.scatter(time, exp_magnesium, label='Experimental magnesium')
# plt.scatter(time, exp_sodium, label='Experimental sodium')
# plt.scatter(time, exp_calcium, label='Experimental calcium')
# plt.scatter(time, exp_glucose, label='Experimental glucose')
# plt.xlabel('Time(min)')
# plt.ylabel('Ct by C0')
# plt.legend()
# plt.show()

#PART 2: ANALYSIS OF GLUCOSE'S DATA 

#PART 2.1: APPLY KINETIC MODELS / FIND MODELS' PARAMETERS 

#Data for glucose (unit: mg/mL)
observed_glucose_2 = np.array([19.26, 1.03, 6.74, 12.30, 17.78, 19.11, 19.31, 19.13])
gl_CtbyC0 = np.array(observed_glucose_2[1:] / observed_glucose_2[0])

#Calculation log(Ct/C0) 
glucose_Th = np.log((1/gl_CtbyC0)-1)
glucose_YN = np.log(1/((1/gl_CtbyC0)-1))

#Drop NaN value 
data_Th = {'x': time, 'y': glucose_Th}
df_gl_Th = pd.DataFrame(data_Th)
df_gl_Th = df_gl_Th.dropna(subset=['y'], how= 'any')

x_time = df_gl_Th.loc[:, ['x']]
y_CtbyC0_Th = df_gl_Th.loc[:, ['y']]

data_YN = {'x': time, 'y': glucose_YN}
df_gl_YN = pd.DataFrame(data_YN)
df_gl_YN = df_gl_YN.dropna(subset=['y'], how= 'any')

x_time = df_gl_YN.loc[:, ['x']]
y_CtbyC0_YN = df_gl_YN.loc[:, ['y']]

# Linear regression for Thomas
x_gl_Th = np.array(x_time).reshape(-1, 1)
y_gl_Th = np.array(y_CtbyC0_Th)

model_gl_Th = LinearRegression()
model_gl_Th.fit(x_gl_Th, y_gl_Th)
model_gl_Th.intercept_  
model_gl_Th.coef_ 

rsq_gl_Th = model_gl_Th.score(x_gl_Th,y_gl_Th)
y_pred_gl_Th = model_gl_Th.predict(x_gl_Th)

# Parameters for Thomas
gl_k_Th = (-1 * model_gl_Th.coef_) / observed_glucose_2[0] #mL/min.mg
gl_q_0 = (model_gl_Th.intercept_ * Q) / (gl_k_Th * m) #mg/g

#Linear regression for Yoon-Nelson model
x_gl_YN = np.array(x_time)
y_gl_YN = np.array(y_CtbyC0_YN)

model_gl_YN = LinearRegression()
model_gl_YN.fit(x_gl_YN, y_gl_YN)
model_gl_YN.intercept_  
model_gl_YN.coef_  

#Parameters for Yoon-Nelson 
rsq_gl_YN = model_gl_YN.score(x_gl_YN,y_gl_YN)
y_pred_gl_YN = model_gl_YN.predict(x_gl_YN)

gl_k_YN = model_gl_YN.coef_ 
gl_torque = model_gl_YN.intercept_ / (-1 * gl_k_YN)

# Table of parameters 
data = [['Thomas', observed_glucose_2[0], Q, gl_k_Th, gl_q_0, rsq_gl_Th], 
         ['Yoon-Nelson', observed_glucose_2[0], Q, gl_k_YN, gl_torque, rsq_gl_YN]]

headers=["Model", "Glucose C0 (mg/mL)", "Q (mL/min)", "Rate constant", "Parameter", "R_squared"]
# print (tabulate(data, headers))

#PART 2.2: PREDICTED Ct/C0 FROM MODELS 

#Plot experimental glucose breakthrough curve - manually drop out the value which give NaN in log calculation 
time_new = np.array([10, 20, 30, 60, 120, 240])
observed_glucose_new = np.array([19.26, 1.03, 6.74, 12.30, 17.78, 19.11, 19.13])
gl_CtbyC0_new = np.array(observed_glucose_new[1:] / observed_glucose_new[0])
plt.scatter(time_new, gl_CtbyC0_new, label='Experimental glucose')

gl_predicted_Th = 1 / (1 + np.exp (gl_k_Th * gl_q_0 * m / Q - gl_k_Th * observed_glucose_2[0] * time_new))
gl_predicted_Th = np.array(gl_predicted_Th).reshape(-1,1)
plt.plot(time_new, gl_predicted_Th, label='Glucose by Thomas')

gl_predicted_YN = 1 / (1 + np.exp(gl_k_YN * (gl_torque - time_new)))
gl_predicted_YN = np.array(gl_predicted_YN).reshape(-1,1)
plt.plot(time_new, gl_predicted_YN, label='Yoon-Nelson')

plt.xlabel('Time(min)')
plt.ylabel('Ct by C0')
plt.legend(loc='lower right')
# plt.show()

#PART 2.3: ERROR ANALYSIS  

#Error analysis for Thomas 
n = len(time_new) #number of data points
p = 2 #number of parameters 
q_e = gl_CtbyC0 #Experimental CtbyC0
q_cal_Th = gl_predicted_Th #Calculated CtbyC0 from model 

#The sum of the squared of the error (ERRSQ) 
sum_squared_errors = 0
for i in range(n):
        error = q_e[i] - q_cal_Th[i]
        squared_error = error ** 2
        sum_squared_errors += np.sum(squared_error)
ERRSQ_Th = sum_squared_errors

#Hybrid fractional error function (HYBRID)
def calculate_hfe(n, p, q_e, q_cal_Th):
        sum_squared_errors = sum(
            ((q_e[i] - q_cal_Th[i])**2 / q_e[i]) for i in range(n))
        return 100 / (n - p) * sum_squared_errors
hfe_Th = calculate_hfe(n, p, q_e, q_cal_Th)

#Marquardt’s percent standard deviation (MPSD)
def calculate_mpsd(n, p, q_e, q_cal_Th):
        # return sum(((df_CtbyC0[i] - Th_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n))
        return 100 * np.sqrt ((1/ (n - p)) * sum(((q_e[i] - q_cal_Th[i]) / q_e[i]) **2 for i in range(n)))
mpsd_Th = calculate_mpsd(n, p, q_e, q_cal_Th)

#PART 3: ANALYSIS OF MAGNESIUM'S DATA 

#Unit conversion 
Mg_molecular_weight = 203.3 #g/mol
observed_magnesium_2 = np.array(observed_magnesium * Mg_molecular_weight / 1000)
Mg_CtbyC0 = np.array(observed_magnesium_2[1:] / observed_magnesium_2[0])

#PART 3.1: APPLY KINETIC MODELS / FIND MODELS' PARAMETERS

#Calculation log(Ct/C0) 
Mg_Th = np.log((1/Mg_CtbyC0)-1)
Mg_YN = np.log(1/((1/Mg_CtbyC0)-1))

# Linear regression for Thomas
x_Mg_Th = np.array(time).reshape(-1, 1)
y_Mg_Th = np.array(Mg_Th)

model_Mg_Th = LinearRegression()
model_Mg_Th.fit(x_Mg_Th, y_Mg_Th)
model_Mg_Th.intercept_  
model_Mg_Th.coef_ 

rsq_Mg_Th = model_Mg_Th.score(x_Mg_Th,y_Mg_Th)
y_pred_Mg_Th = model_Mg_Th.predict(x_Mg_Th)

# Parameters for Thomas
Mg_k_Th = (-1 * model_Mg_Th.coef_) / observed_magnesium_2[0] #mL/min.mg
Mg_q_0 = (model_Mg_Th.intercept_ * Q) / (Mg_k_Th * m) #mg/g

#Linear regression for Yoon-Nelson model
x_Mg_YN = np.array(time).reshape(-1, 1)
y_Mg_YN = np.array(Mg_YN)

model_Mg_YN = LinearRegression()
model_Mg_YN.fit(x_Mg_YN, y_Mg_YN)
model_Mg_YN.intercept_  
model_Mg_YN.coef_  

#Parameters for Yoon-Nelson 
rsq_Mg_YN = model_Mg_YN.score(x_Mg_YN,y_Mg_YN)
y_pred_Mg_YN = model_Mg_YN.predict(x_Mg_YN)

Mg_k_YN = model_Mg_YN.coef_ 
Mg_torque = model_Mg_YN.intercept_ / (-1 * Mg_k_YN)

#PART 3.2: PREDICTED Ct/C0 FROM MODELS 

#PART 3.3: ERROR ANALYSIS

#PART 4: ANALYSIS OF CALCIUM'S DATA

#Unit conversion 
Ca_molecular_weight = 111.56 #g/mol
observed_calcium_2 = np.array(observed_calcium * Ca_molecular_weight / 1000)
Ca_CtbyC0 = np.array(observed_calcium_2[1:] / observed_calcium_2[0])

#PART 4.1: APPLY KINETIC MODELS / FIND MODELS' PARAMETERS

#PART 4.2: PREDICTED Ct/C0 FROM MODELS 

#PART 4.3: ERROR ANALYSIS

