import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, interpolate
import random



def objective(k_Th, c_ddr, q_e, Q):
    V_waste = 0 #ml
    # c_w = pd.DataFrame({'waste': [0]*241}) #mg/mL 
    c_w = pd.DataFrame({'waste': [0]*281}) #mg/mL 
    # glucose_mass = np.zeros(241)
    glucose_mass = np.zeros(481)
    s = [0]

    # for t in range(1, 241):
    for t in range (1, 481):
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
# print(df)

t = 0
x_AC = 200 #g 
c_ddr = 19.26 #mg/mL
k_Th = 0.00155042 #ml/min/mg
q_e = 159.049 #mg/g 
Q = 60 #mL/min

# c_w, s, glucose_mass = objective(k_Th, c_ddr, q_e, Q)
# glucose_mass = glucose_mass / 1000

# print(c_w, s, glucose_mass)

# xnew = np.linspace(0, 240, num=241)
xnew = np.linspace(0, 480, num=481)

# #Plot
# fig, ax = plt.subplots(1,2)
# ax[0].scatter(df.index, df['Waste reservoir'], label = 'Experimental waste')
# c_w.plot(ax = ax[0], c = 'b')
# ax[1].scatter(df.index, df['Downstream sorbent'], c = 'k', label = 'Experimental sorbents')
# ax[1].plot(xnew, s, c= 'k', label = 'Downstream sorbent')
# ax[0].legend()
# plt.legend()

# plt.figure()
# plt.plot(xnew, glucose_mass) 
# plt.xlabel('Time (min)')
# plt.ylabel('Glucose Mass (g)')
# plt.show()

#Total glucose removal at 180 mins - cummulative sum of all glucose removed
# total_glucose_mass_removed = sum(glucose_mass[:480])
# print(total_glucose_mass_removed)



'''
EVALUATE THE INFLUENCE OF PARAMETERS

The function relationship between k_Th and flow rate: Quadratic function a*x**2 + b*x +c with 
a = 2.0520839379638302e-05 
b = -0.3868455231248755
c = 20.57987780585551

Amount of total glucose removed in initial data: 45.508613359381606 (g) with Q = 60 mL/min

'''

#PART 1: ADJUST FLOW RATE (original Q = 60 mL/min) - adsorption capacity q_e is assumed to be constant

#Calculate new k_Th which is dependent on flow rate

def calculate_y(x, function):
    return function(x)

def quadratic_function(x):
    a = 4.50018042e-03 
    b = -7.15357057e-01
    c = 2.85623076e+01
    return a*x**2 + b*x +c

# Generate random values for flow rate and calculate its corresponding k_Th

x_values = []

while len(x_values) < 10:
    x = random.randint(50, 150)
    if (x % 10 == 0 or x % 5 == 0) and x not in x_values:
        x_values.append(x)
x_values.sort()

y_values = [calculate_y(x, quadratic_function) for x in x_values]
# for i, x in enumerate(x_values):
    # print(f"x_{i+1} = {x:.2f}, y_{i+1} = {y_values[i]:.3f}")

t = 0
x_AC = 200 #g 
c_ddr = 19.26 #mg/mL
q_e = 159.049 #mg/g 

total_glucose_removed_list_1 = []

for Q, k_Th in zip(x_values, y_values):
    c_w_new, s_new, glucose_mass_new = objective(k_Th, c_ddr, q_e, Q)
    # print(f"The result for Q={Q} and k_Th={k_Th} is {c_w_new, s_new, glucose_mass_new}")
    glucose_mass_new = glucose_mass_new / 1000
    total_glucose_removed = sum(glucose_mass_new[:480])
    total_glucose_removed_list_1.append(total_glucose_removed)
    # print(f"Total glucose mass removed for flow rate={Q} and k_Th={k_Th} is {total_glucose_removed}")
    
#Plot to see the trend
plt.figure()
plt.scatter(x_values, total_glucose_removed_list_1) 
plt.xlabel('Flow Rate')
plt.ylabel('Total glucose removed (g)')
# plt.show()



#PART 2: ADJUST ADSORPTION CAPACITY (q_e) - keep k_Th and Q constant 

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
    # print(f"The result for q_e={q_e} is {c_w_new, s_new, glucose_mass_new}")
    glucose_mass_new = glucose_mass_new / 1000
    total_glucose_removed = sum(glucose_mass_new[:480])
    total_glucose_removed_list_2.append(total_glucose_removed)
    # print(f"For q_e = {q_e}, total glucose removed = {total_glucose_removed}")

#Plot to see the trend
plt.figure()
plt.scatter(q_e_list, total_glucose_removed_list_2) 
plt.xlabel('Adsorption capacity')
plt.ylabel('Total glucose removed (g)')
# plt.show()



#PART 3: ADJUST THE MASS OF SORBENT - keep q_e and k_Th constant 

x_AC_values = [random.uniform(100, 500) for _ in range(10)]
x_AC_values.sort()

total_glucose_removed_list_3 = []

for x_AC in x_AC_values:
    c_w_new, s_new, glucose_mass_new = objective(k_Th, c_ddr, q_e, Q)
    # print(f"The result for x_AC={x_AC} is {c_w_new, s_new, glucose_mass_new}")
    glucose_mass_new = glucose_mass_new / 1000
    total_glucose_removed = sum(glucose_mass_new[:480])
    total_glucose_removed_list_3.append(total_glucose_removed)
    print(f"For x_AC = {x_AC}, total glucose removed = {total_glucose_removed}")

#Plot to see the trend
plt.figure()
plt.scatter(x_AC_values, total_glucose_removed_list_3) 
plt.xlabel('Mass of the sorbent')
plt.ylabel('Total glucose removed (g)')
plt.show()

