import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
from scipy.optimize import curve_fit



'''Plotting of k_Th vs. flow rate''' 
file_path = 'fitting_thomas_model_parameters.xlsx'
sheet_name = 'Sheet2'
df = pd.read_excel(file_path, sheet_name=sheet_name, header = None, names=['Flow Rate', 'k_Th', 'q_e'])
# df.plot.scatter('flow rate', 'k_Th')
# plt.show()

FlowRate = df['Flow Rate']
RateConstant = df['k_Th']
avg_kTh = df.groupby('Flow Rate')['k_Th'].transform('mean')



'''Curve-fitting'''

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



# Perform curve fitting

# y-values as k_Th

# functions = [func1, func2, func3, func4, func5, func6]
# r_sq = []
# popts = []
# pcovs = []

# for i, func in enumerate([func1, func2, func3, func4, func5, func6]):
#     popt, pcov = curve_fit(func, FlowRate, RateConstant, maxfev=10000)
#     residuals = RateConstant - func(FlowRate, *popt)
#     ss_res = np.sum(residuals ** 2)
#     ss_tot = np.sum((RateConstant-np.mean(RateConstant))**2)
#     r_sq_i = 1 - (ss_res / ss_tot)
#     r_sq.append(r_sq_i)
#     popts.append(popt)
#     pcovs.append(pcov)
#     print(f"R-squared for func{i+1}: {r_sq_i}")

# y-values as avg_kTh

functions = [func1, func2, func3, func4, func5, func6]
r_sq = []
popts = []
pcovs = []

for i, func in enumerate([func1, func2, func3, func4, func5, func6]):
    popt, pcov = curve_fit(func, FlowRate, avg_kTh, maxfev=10000)
    residuals = avg_kTh - func(FlowRate, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((avg_kTh - np.mean(avg_kTh)) ** 2)
    r_sq_i = 1 - (ss_res / ss_tot)
    r_sq.append(r_sq_i)
    popts.append(popt)
    pcovs.append(pcov)
    print(f"R-squared for func{i+1}: {r_sq_i}")



#Determine the best fit function 
best_fit = np.argmax(r_sq)
print(f"\nThe best fit is func{best_fit+1} with R-squared value of {r_sq[best_fit]}")



# Plot

# #y-values as Rate Constant 
# plt.plot(FlowRate, RateConstant, 'o', label='Original data')

# labels = [f'Fitted curve func{i}' for i in range(len(functions))]

# for i, func in enumerate(functions[:5]):
#     q_fit = np.linspace(FlowRate.min(), FlowRate.max(), 100)
#     k_Th_fit = func(q_fit, *popts[i])
#     plt.plot(q_fit, k_Th_fit, label=labels[i])

# #Only for the last function - function 6
# q_fit = np.linspace(FlowRate.min(), FlowRate.max(), 100)
# k_Th_fit = func6(q_fit, *popts[5])
# plt.plot(q_fit, k_Th_fit, 'b-', label='Fitted curve func6')

#y-values as avg_kTh
plt.plot(FlowRate, avg_kTh, 'o', label='Original data')

labels = [f'Fitted curve func{i}' for i in range(len(functions))]

for i, func in enumerate(functions[:5]):
    q_fit = np.linspace(FlowRate.min(), FlowRate.max(), 100)
    k_Th_fit = func(q_fit, *popts[i])
    plt.plot(q_fit, k_Th_fit, label=labels[i])

#Only for the last function - function 6
q_fit = np.linspace(FlowRate.min(), FlowRate.max(), 100)
k_Th_fit = func6(q_fit, *popts[5])
plt.plot(q_fit, k_Th_fit, 'b-', label='Fitted curve func6')

plt.legend()
plt.xlabel('Flow rate')
plt.ylabel('k_Th')
# plt.show()



# Print the best-fit parameters
best_fit_index = r_sq.index(max(r_sq))
best_fit_params = popts[best_fit_index]
print(f"Parameters of best fit function: {best_fit_params}")