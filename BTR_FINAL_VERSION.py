import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
import scipy
import random
from scipy.optimize import curve_fit

#RECOMMENDATION: Run each region separately 

#region PART 1: PLOT EXPERIMENTAL BREAKTHROUGH CURVES

#Time (mins)
time = np.array([10, 20, 30, 60, 120, 180, 240])

#Flow rate (mL/min)
Q = 60 

#Mass of the adsorbent - activated carbon (g)
m = 200 

#Effluent concentration Ct (mmol/L) - the first value is influent concentration C0
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

#Plot to see the curve
plt.scatter(time, exp_bicarbonate, label='Experimental bicarbonate')
plt.scatter(time, exp_magnesium, label='Experimental magnesium')
plt.scatter(time, exp_sodium, label='Experimental sodium')
plt.scatter(time, exp_calcium, label='Experimental calcium')
plt.scatter(time, exp_glucose, label='Experimental glucose')
plt.xlabel('Time(min)')
plt.ylabel('Ct by C0')
plt.legend()
plt.show()

#endregion

#region PART 2: APPLY KINETIC MODELS and FIND MODELS' PARAMETERS 

#GLUCOSE 

#Data for glucose (unit: mg/mL) - the first value is C0, others are Ct
observed_glucose_2 = np.array([19.26, 1.03, 6.74, 12.30, 17.78, 19.11, 19.31, 19.13])

#Ct/C0
gl_CtbyC0 = np.array(observed_glucose_2[1:] / observed_glucose_2[0])

#Calculation log(Ct/C0) 
glucose_Th = np.log((1/gl_CtbyC0)-1) #for Thomas model
glucose_YN = np.log(1/((1/gl_CtbyC0)-1)) #for Yoon-Nelson model

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

#Linear regression for Thomas using sklearn.linear_model
x_gl_Th = np.array(x_time).reshape(-1, 1)
y_gl_Th = np.array(y_CtbyC0_Th)

model_gl_Th = LinearRegression()
model_gl_Th.fit(x_gl_Th, y_gl_Th)
model_gl_Th.intercept_  
model_gl_Th.coef_ 

rsq_gl_Th = model_gl_Th.score(x_gl_Th,y_gl_Th)
y_pred_Gl_Th = model_gl_Th.predict(x_gl_Th)

#Parameters for Thomas
gl_k_Th = (-1 * model_gl_Th.coef_) / observed_glucose_2[0]
gl_q_0 = (model_gl_Th.intercept_ * Q) / (gl_k_Th * m)

#Linear regression for Yoon-Nelson model using sklearn.linear_model
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
print (tabulate(data, headers))

#Predicted Ct/C0 based on models

#Plot experimental glucose breakthrough curve - manually drop out the value which give NaN in log calculation
#Plot this the second time because of using different unit for concentration 
time_new = np.array([10, 20, 30, 60, 120, 240]) #drop 180mins because there is NaN value detected at this time point 
observed_glucose_new = np.array([19.26, 1.03, 6.74, 12.30, 17.78, 19.11, 19.13]) #drop the corresponded value of 180mins
gl_CtbyC0_new = np.array(observed_glucose_new[1:] / observed_glucose_new[0]) 
plt.scatter(time_new, gl_CtbyC0_new, label='Experimental data')

#Ct/C0 predicted from the Thomas model
gl_predicted_Th = 1 / (1 + np.exp (gl_k_Th * gl_q_0 * m / Q - gl_k_Th * observed_glucose_2[0] * time_new))
gl_predicted_Th = np.array(gl_predicted_Th).reshape(-1,1)

#Ct/C0 predicted from the Yoon-Nelson model
gl_predicted_YN = 1 / (1 + np.exp(gl_k_YN * (gl_torque - time_new)))
gl_predicted_YN = np.array(gl_predicted_YN).reshape(-1,1)

#Plotting
plt.plot(time_new, gl_predicted_Th, label='Thomas')
plt.plot(time_new, gl_predicted_YN, label='Yoon-Nelson')
plt.xlabel('Time(min)')
plt.ylabel('Ct by C0')
plt.legend(loc='lower right')
plt.show()

#Error analysis for Thomas 
n = len(time_new) #number of data points
p = 2 #number of parameters 
q_e = gl_CtbyC0 #Experimental CtbyC0
q_cal_Th = gl_predicted_Th #Calculated CtbyC0 from model 

#endregion

#region PART 3: CALCULATION OF SOLUTE CONCENTRATION IN DOWNSTREAM SORBENT AND WASTE RESERVOIR

def objective(k_Th, c_ddr, q_e, Q):
    #Initialise the volume of the waste reservoir
    V_waste = 0 #ml
        
    #Create an empty DataFrame for waste reservoir concentration
    #if use total time = 240 mins
    c_w = pd.DataFrame({'Calculated waste': [0]*241}) #mg/mL 
    glucose_mass = np.zeros(241)
    #if use total time = 480 mins
    # c_w = pd.DataFrame({'waste': [0]*481}) #mg/mL 
    # glucose_mass = np.zeros(481)
    
    #Create an empty DataFrame for downstream sorbent concentration
    s = [0]

    for t in range(1, 241):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/Q-k_Th*c_ddr*t))) #deltaT = 1 mg/ml 
        glucose_mass[t] = (c_ddr - c_ds) * Q * 1         
        s.append(c_ds)    
        c_w.loc[t] = (Q*c_ds*1+V_waste*c_w.loc[t-1])/(Q + V_waste) #mg/mL
        V_waste += Q * 1 #ml
    return c_w, s, glucose_mass

file_path = '20230418 Experiment single pass.xlsx'
sheet_name = 'Sheet1'
rows = np.array(range(1, 8))
cols = [0, 1, 2, 3]
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=cols, header = 0)
df.columns = ['Time', 'Downstream dialysate reservoir', 'Downstream sorbent', 'Waste reservoir']
df.set_index('Time', inplace=True)
print(df)

#Provide these values for the function to perform 
t = 0
x_AC = 200 #g 
c_ddr = 19.26 #mg/mL
k_Th = 0.00155042 #ml/min/mg
q_e = 159.049 #mg/g
Q = 60 #mL/min

c_w, s, glucose_mass = objective(k_Th, c_ddr, q_e, Q)
glucose_mass = glucose_mass / 1000
# print(c_w, s, glucose_mass)

#Return evenly spaced numbers over a specified time interval
xnew = np.linspace(0, 240, num=241)
# xnew = np.linspace(0, 480, num=481)

#Plotting
fig, ax = plt.subplots(1,2, figsize = (12, 8))
df['Waste reservoir'] = df['Waste reservoir'].iloc[2:]
df['Downstream sorbent'] = df['Downstream sorbent'].iloc[2:]
ax[0].scatter(df.index, df['Waste reservoir'], label = 'Experimental waste')
c_w.plot(ax = ax[0], c = 'b')
ax[0].set_xlabel('Time (min)', fontsize = 18)
ax[0].set_ylabel('Glucose concentration (mg/mL)', fontsize = 18)
ax[0].legend(fontsize = 12)
ax[1].scatter(df.index, df['Downstream sorbent'], c = 'k', label = 'Experimental downstream sorbent')
ax[1].plot(xnew, s, c= 'k', label = 'Calculated downstream sorbent')
ax[1].set_xlabel('Time (min)', fontsize = 18)
ax[1].set_ylabel('Glucose concentration (mg/mL)', fontsize = 18)
plt.legend(fontsize = 12)

plt.figure()
plt.plot(xnew, glucose_mass) 
plt.xlabel('Time (min)')
plt.ylabel('Glucose mass (g)')
plt.show()

#Total glucose removal at 240 mins - cumulative sum of all glucose removed
total_glucose_mass_removed = sum(glucose_mass[:240])
# total_glucose_mass_removed = sum(glucose_mass[:480])
print(total_glucose_mass_removed)

#endregion

#region PART 4: FINDING THE FUNCTION THAT REPRESENT THE RELATIONSHIP BETWEEN THOMAS RATE CONSTANT AND FLOW RATE 

#Plotting of k_Th vs. flow rate 
file_path = 'fitting_thomas_model_parameters.xlsx'
sheet_name = 'Sheet2'
df = pd.read_excel(file_path, sheet_name=sheet_name, header = None, names=['Flow Rate', 'k_Th', 'q_e'])

#Plot to see the trend 
# df.plot.scatter('flow rate', 'k_Th')
# plt.show()

FlowRate = df['Flow Rate']
RateConstant = df['k_Th']

#Define function
def func1(x, a, b, c):
    return a + b*x + c*np.exp(-x)
def func2(x, a, b, c):
    return a*x**2 + b*x +c
def func3(x, a, b, c):
    return a*x**3 + b*x + c
def func4(x, a, b, c):
    return a*x**3 + b*x**2 + c
def func5(x, a, b):
    return a*x + b  
def func6(x, a, b):
    return a * np.power(x, b)

#Perform curve fitting
functions = [func1, func2, func3, func4, func5, func6]
r_sq = []
popts = []
pcovs = []

for i, func in enumerate([func1, func2, func3, func4, func5, func6]):
    popt, pcov = curve_fit(func, FlowRate, RateConstant, maxfev=10000)
    residuals = RateConstant - func(FlowRate, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((RateConstant-np.mean(RateConstant))**2)
    r_sq_i = 1 - (ss_res / ss_tot)
    r_sq.append(r_sq_i)
    popts.append(popt)
    pcovs.append(pcov)
    print(f"R-squared for func{i+1}: {r_sq_i}")

#Determine the best fit function 
best_fit = np.argmax(r_sq)
print(f"\nThe best fit is func{best_fit+1} with R-squared value of {r_sq[best_fit]}")

#Plot
plt.plot(FlowRate, RateConstant, 'o', label='Original data')
labels = [f'Fitted curve func{i}' for i in range(len(functions))]

#Label
for i, func in enumerate(functions[:5]):
    q_fit = np.linspace(FlowRate.min(), FlowRate.max(), 100)
    k_Th_fit = func(q_fit, *popts[i])
    plt.plot(q_fit, k_Th_fit, label=labels[i+1])

#Only for the last function - function 6
q_fit = np.linspace(FlowRate.min(), FlowRate.max(), 100)
k_Th_fit = func6(q_fit, *popts[5])
plt.plot(q_fit, k_Th_fit, 'b-', label='Fitted curve func6')

plt.legend()
plt.xlabel('Flow rate')
plt.ylabel('k_Th')
plt.show()

#Print the best-fit parameters
best_fit_index = r_sq.index(max(r_sq))
best_fit_params = popts[best_fit_index]
print(f"Parameters of best fit function: {best_fit_params}")

#endregion

#region PART 5: EVALUATE THE INFLUENCE OF PARAMETERS - FLOW RATE

'''
Information:

THIS REGION ADJUST FLOW RATE (original Q = 60 mL/min) - adsorption capacity q_e is assumed to be constant

The function relationship between k_Th and flow rate: Quadratic function a*x**2 + b*x +c with 
a = 2.0520839379638302e-05 
b = -0.3868455231248755
c = 20.57987780585551

Amount of total glucose removed in initial data: 45.508613359381606 (g) with Q = 60 mL/min
'''

#Calculate new k_Th which is dependent on flow rate - y = k_Th and x = Q 
def calculate_y(x, function):
    return function(x)
def quadratic_function(x):
    a = 4.50018042e-03 
    b = -7.15357057e-01
    c = 2.85623076e+01
    return a*x**2 + b*x +c

#Generate random values for flow rate and calculate its corresponding k_Th - 10 values in total 
x_values = []
while len(x_values) < 10:
    x = random.randint(50, 150)
    if (x % 10 == 0 or x % 5 == 0) and x not in x_values:
        x_values.append(x)
x_values.sort()

y_values = [calculate_y(x, quadratic_function) for x in x_values]

#Conditions for the simulation
t = 0
x_AC = 200 #g 
c_ddr = 19.26 #mg/mL
q_e = 159.049 #mg/g 

total_glucose_removed_list_1 = []

for Q, k_Th in zip(x_values, y_values):
    c_w_new, s_new, glucose_mass_new = objective(k_Th, c_ddr, q_e, Q)
    # print(f"The result for Q={Q} and k_Th={k_Th} is {c_w_new, s_new, glucose_mass_new}")
    glucose_mass_new = glucose_mass_new / 1000
    total_glucose_removed = sum(glucose_mass_new[:240])
    total_glucose_removed_list_1.append(total_glucose_removed)
    # print(f"Total glucose mass removed for flow rate={Q} and k_Th={k_Th} is {total_glucose_removed}")

#Plot to see the trend
plt.figure()
plt.scatter(x_values, total_glucose_removed_list_1)
plt.xlabel('Flow rate (mL/min)')
plt.ylabel('Total glucose removed (g)')

#Experimental baseline
baseline1 = 60
plt.axvline(x=baseline1, color='r', linestyle='--', label='Baseline')

#Run the objective function to calculate the corresponded y-line 
def objective(k_Th, c_ddr, q_e, Q):
    V_waste = 0 #ml
    c_w = pd.DataFrame({'Calculated waste': [0]*241}) #mg/mL 
    glucose_mass0 = np.zeros(241)
    # c_w = pd.DataFrame({'waste': [0]*281}) #mg/mL 
    # glucose_mass = np.zeros(481)
    s = [0]
    
    t = 0
    x_AC = 200 #g 
    c_ddr = 19.26 #mg/mL
    k_Th = 1.842 #ml/min/mg
    q_e = 159.049 #mg/g 
    Q = 60 #mL/min

    for t in range(1, 241):
    # for t in range (1, 481):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/Q-k_Th*c_ddr*t))) #deltaT = 1 mg/ml 
        glucose_mass0[t] = (c_ddr - c_ds) * Q * 1         
        s.append(c_ds)    
        c_w.loc[t] = (Q*c_ds*1+V_waste*c_w.loc[t-1])/(Q + V_waste) #mg/mL
        V_waste += Q * 1 #ml
    return glucose_mass0

glucose_mass0 = objective(k_Th, c_ddr, q_e, Q)
glucose_mass0 = glucose_mass0 / 1000
baseline2 = sum(glucose_mass0[:240])
plt.axhline(y=baseline2, color='r', linestyle='--', label='Baseline')

plt.show()

#endregion 

#region PART 6: EVALUATE THE INFLUENCE OF PARAMETERS - ADSORPTION CAPACITY

#THIS REGION ADJUST ADSORPTION CAPACITY (q_e) - keep k_Th and Q constant 

#Conditions for the simulation
t = 0
x_AC = 200 #g 
c_ddr = 19.26 #mg/mL
k_Th = 0.00155042 #ml/min/mg
Q = 60 #mL/min

q_e_list = [random.uniform(80, 320) for _ in range(10)]
q_e_list.sort()
total_glucose_removed_list_2 = []
for q_e in q_e_list:
    c_w_new, s_new, glucose_mass_new = objective(k_Th, c_ddr, q_e, Q)
    print(f"The result for q_e={q_e} is {c_w_new, s_new, glucose_mass_new}")
    glucose_mass_new = glucose_mass_new / 1000
    total_glucose_removed = sum(glucose_mass_new[:240])
    total_glucose_removed_list_2.append(total_glucose_removed)
    print(f"For q_e = {q_e}, total glucose removed = {total_glucose_removed}")

#Plot to see the trend
plt.figure()
plt.scatter(q_e_list, total_glucose_removed_list_2) 
plt.xlabel('Adsorption capacity (mg/g)')
plt.ylabel('Total glucose removed (g)')

#Experimental baseline
baseline1 = 159.049
plt.axvline(x=baseline1, color='r', linestyle='--', label='Baseline')

def objective(k_Th, c_ddr, q_e, Q):
    V_waste = 0 #ml
    c_w = pd.DataFrame({'Calculated waste': [0]*241}) #mg/mL 
    glucose_mass0 = np.zeros(241)
    # c_w = pd.DataFrame({'waste': [0]*281}) #mg/mL 
    # glucose_mass = np.zeros(481)
    s = [0]
    
    t = 0
    x_AC = 200 #g 
    c_ddr = 19.26 #mg/mL
    k_Th = 0.00155042 #ml/min/mg
    q_e = 159.049 #mg/g 
    Q = 60 #mL/min

    for t in range(1, 241):
    # for t in range (1, 481):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/Q-k_Th*c_ddr*t))) #deltaT = 1 mg/ml 
        glucose_mass0[t] = (c_ddr - c_ds) * Q * 1         
        s.append(c_ds)    
        c_w.loc[t] = (Q*c_ds*1+V_waste*c_w.loc[t-1])/(Q + V_waste) #mg/mL
        V_waste += Q * 1 #ml
    return glucose_mass0

glucose_mass0 = objective(k_Th, c_ddr, q_e, Q)
glucose_mass0 = glucose_mass0 / 1000
baseline2 = sum(glucose_mass0[:240])
print(baseline2)
plt.axhline(y=baseline2, color='r', linestyle='--', label='Baseline')

plt.show()

#endregion

#region PART 7: EVALUATE THE INFLUENCE OF PARAMETERS - SORBENT MASS 

x_AC_values = [random.uniform(100, 500) for _ in range(10)]
x_AC_values.sort()

#Conditions for the simulation 
t = 0
c_ddr = 19.26 #mg/mL
k_Th = 0.00155042 #ml/min/mg
q_e = 159.049 #mg/g 
Q = 60 #mL/min

total_glucose_removed_list_3 = []

for x_AC in x_AC_values:
    c_w_new, s_new, glucose_mass_new = objective(k_Th, c_ddr, q_e, Q)
    # print(f"The result for x_AC={x_AC} is {c_w_new, s_new, glucose_mass_new}")
    glucose_mass_new = glucose_mass_new / 1000
    total_glucose_removed = sum(glucose_mass_new[:240])
    total_glucose_removed_list_3.append(total_glucose_removed)
    print(f"For x_AC = {x_AC}, total glucose removed = {total_glucose_removed}")

#Plot to see the trend
plt.figure()
plt.scatter(x_AC_values, total_glucose_removed_list_3) 
plt.xlabel('Mass of the sorbent (g)')
plt.ylabel('Total glucose removed (g)')

#Experimental baseline
baseline1 = 200
plt.axvline(x=baseline1, color='r', linestyle='--', label='Baseline')

def objective(k_Th, c_ddr, q_e, Q):
    V_waste = 0 #ml
    c_w = pd.DataFrame({'Calculated waste': [0]*241}) #mg/mL 
    glucose_mass0 = np.zeros(241)
    # c_w = pd.DataFrame({'waste': [0]*481}) #mg/mL 
    # glucose_mass = np.zeros(481)
    s = [0]
    
    t = 0
    x_AC = 200 #g 
    c_ddr = 19.26 #mg/mL
    k_Th = 0.00155042 #ml/min/mg
    q_e = 159.049 #mg/g 
    Q = 60 #mL/min

    for t in range(1, 241):
    # for t in range (1, 481):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/Q-k_Th*c_ddr*t))) #deltaT = 1 mg/ml 
        glucose_mass0[t] = (c_ddr - c_ds) * Q * 1         
        s.append(c_ds)    
        c_w.loc[t] = (Q*c_ds*1+V_waste*c_w.loc[t-1])/(Q + V_waste) #mg/mL
        V_waste += Q * 1 #ml
    return glucose_mass0

glucose_mass0 = objective(k_Th, c_ddr, q_e, Q)
glucose_mass0 = glucose_mass0 / 1000
baseline2 = sum(glucose_mass0[:240])
print(baseline2)
plt.axhline(y=baseline2, color='r', linestyle='--', label='Baseline')

plt.show()

#endregion

#region PART 8: OPTIMIZATION 
#Find the best combination of parameters for maximum amount of total glucose removed 

def optimize(x):
    print(x)
    V_waste = 0 #ml
    c_w = pd.DataFrame({'waste': [0]*241}) #mg/mL 
    glucose_mass = np.zeros(241)
    s = [0]
    k_Th = quadratic_function(x[1])
    Q = x[1] 
    x_AC = x[0]
    q_e = x[2]
    c_ddr = 19.26

    for t in range(1, 241):
    # for t in range (1, 481):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/Q-k_Th*c_ddr*t))) #deltaT = 1 mg/ml 
        glucose_mass[t] = (c_ddr - c_ds) * Q * 1          
        c_w.loc[t] = (Q*c_ds*1+V_waste*c_w.loc[t-1])/(Q + V_waste) #mg/mL
        V_waste += Q * 1 #ml
    total_glucose_mass_removed = sum(glucose_mass[:240])
    return -total_glucose_mass_removed

#Set the range for each parameter 
Q_bounds = (50, 250)
q_e_bounds = (80, 320)
x_AC_bounds = (100, 500)  

bounds = [x_AC_bounds, Q_bounds, q_e_bounds]  # Combine the bounds

x_val = []
obj_fn = []

for _ in range(10):
    x0 = random.randrange(x_AC_bounds[0], x_AC_bounds[1])
    x1 = random.randrange(50, 250)
    x2 = random.randrange(q_e_bounds[0], q_e_bounds[1])
    initial_guess = [x0, x1, x2]

    result = scipy.optimize.minimize(optimize, initial_guess, method='SLSQP', 
    bounds=bounds, options = {"maxiter" : 1000, "disp": True})
    x_val.append(result['x'])
    obj_fn.append(result['fun'])

#Select the fitted values with the least error
x_sel = x_val[np.argmin(obj_fn)]
#Run the objective subroutine 
glucose_mass = -optimize(x_sel)

#Retrieve the optimal values
optimal_values = result.x
max_glucose_mass = -result.fun
glucose_mass = glucose_mass / 1000

print("Optimal values:")
print("x_AC:", x_sel[0])
print("Q:", x_sel[1])
print("q_e:", x_sel[2])
print("Maximum glucose mass:", glucose_mass)

#endregion 