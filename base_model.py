# -*- coding: utf-8 -*-
"""
Created on Mon May  1 10:11:55 2023

@author: P70073624
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, interpolate

'''
Assumptions: 
    1. I have assumed one q_e value for all flow rates
    2. my code simulates downstream dialysate reservoir -> downstream sorbents -> waste.
'''

def objective(x0, c_ddr, df, f_avg):
    
    '''this is for the prediction sub routine
    to calculate waste reservoir concentration 
    and downstream sorbents concentration. It is the 
    same as the optimisation - so find explanation there'''
  
    V_waste = 0 #ml
    c_w = pd.DataFrame({'waste': [0]*481})
    my_dict = {uniq_f[i]: x0[i] for i in range(len(uniq_f))}
    # q_e = x0[-1]  -> uncomment if you want to fit q_e also
    s = [0]
    
    for t in range(1, 481):
        
        k_Th = my_dict[f_avg[t]] #ml/min/mg
    
        c_ds = c_ddr[t]*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg[t]-k_Th*c_ddr[t]*t))) #deltaT = 1 mg/ml    
               
        s.append(c_ds)
        
        c_w.loc[t] = (f_avg[t]*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg[t] + V_waste) #mg/mL
        
        V_waste += f_avg[t] * 1 #ml
        
        if t % 120 == 0:
            print(V_waste)
            V_waste = 0
        
    return c_w, s


def optimise_fn(x0, c_ddr, df, f_avg):
    
    '''this is the optimisation sub routine.
    Returns- a single value for sum of the error
    '''
    #print(x0)
    
    #initialise the volume of the waste reservoir
    V_waste = 0 #ml
    
    # create an empty dataframe for waste reservoir concentration
    c_w = pd.DataFrame({'waste': [0]*481})
    
    # create an empty dataframe for downstream sorbent concentration
    s = pd.DataFrame({'sorbents': [0]*481})
    
    # create a dictionary to link the different flow rates, Q used in the 
    # experiments to a different k_Th
    my_dict = {uniq_f[i]: x0[i] for i in range(len(uniq_f))}
    
    # Uncomment below if you want to fit q_e
    # q_e = x0[-1]
    
    # start loop to calculate c_ds (conc downstream sorbents) and c_w (concnetration waste)
    for t in range(1, 481):
        
        # set K_Th to the randomly assigned value according to this time step's flow rate, f_avg
        k_Th = my_dict[f_avg[t]] #ml/min/mg
    
        # Use Thomas model to calculate c_ds
        c_ds = c_ddr[t]*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg[t]-k_Th*c_ddr[t]*t))) #deltaT = 1 mg/ml
        
        # append c_ds to s
        s.loc[t] = c_ds
        
        #update c_w  
        '''
        Mass conservation
        mass in the waste container + mass from flow in = final mass in the waste container
        V_waste* cw [t-1] + f_avg [t]*c_ds * timestep = (V_waste + f_avg * timestep)* cfinal
        -> cfinal = (V_waste * cw[t-1] + f_avg[t]*c_ds)/(f_avg[t] + waste) 
        assuming a time step of 1 min
         '''
        c_w.loc[t] = (f_avg[t]*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg[t] + V_waste) #mg/mL
        
        # update waste reservoir volume
        V_waste += f_avg[t] * 1 #ml
        
        #experiments informs that the waste reservoir is changed every 2 hours or 120 minutes
        if t % 120 == 0:
            V_waste = 0
        
    # here you can decide what you want to minimise.
    # I have chosen to minimise the difference in predicted and exptal waste conc
    # since those values are present for all experiments.
    return sum(np.sqrt((df['waste']-c_w.loc[df.index, 'waste'])**2))#+ np.sqrt((df['downstream sorbents']-s.loc[df.index, 'sorbents'])**2))

# Set the file path and sheet name
file_path = 'Copy of Kopie van WEAKID in vitro single pass_V2.xlsx'
sheet_name = '20190806_2'

# Set the desired rows and columns
rows = np.array(range(140,151))  # Rows to read (0-based index) -> glucose
cols = [0, 4, 6, 7, 8, 9]  # Columns to read (0-based index) 

# Load the Excel file into a pandas dataframe
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=cols, header = None)

# Select only the desired rows
df = df.iloc[rows].fillna(0)
df.columns = ['Time', 'Flow rate', 'upstream dialysate reservoir', 'downstream DR', 'downstream sorbents', 'waste']

#change mmol/L to mg/ml
df['waste'] *= (180.156/1000)
df['downstream sorbents'] *= (180.156/1000)
df['downstream DR'] *= (180.156/1000)
df.iloc[0,2] = 0
df.set_index('Time', inplace=True)
# Print the resulting dataframe
print(df)

t = 0
x_AC = 200 #g of activated carbon
q_e = 159.049 #mg/g from Utrecht expts 19.2mg glucose/200g activated carbon
time = df.index

#interpolate flow velocity
xnew = np.linspace(0, 480, num=481)
interp = interpolate.interp1d(df.index, df.iloc[:,0], kind = "next")
f_avg = interp(xnew)*1000

#interpolate downstream dialysate reservoir concentration
c_ddr = np.interp(xnew, time, df.iloc[:,2]) #molecular mass of glucose #mg/mL

# find the unique flow velocities used in the experiment
uniq_f = list(set(f_avg))

# initialise two lists to collect the fitted values and the objective function, i.e., the sum of error
x_val = []
obj_fn = []

# I run each simulation 10 times to avoid getting stuck in local minima
for var in range(10):
    #Define initial initial_guess. Add 1 to len(uniq_f) if you want to also fit q_e
    x0 = np.random.random(len(uniq_f))
    
    '''SLSQP optimisation
    optimise function = the function to be minimised. This has to be a single number - in 
    our case, the sum of error
    x0 = initial guess
    args = extra variables that mightbe needed by the function
    method = sequential least square. there are others. check it out on scipy's website
    bounds = it will try to always keep x0 between these bounds. Since k_Th and q_e is positive,
    it is bound to 0 and inf
    options --
    maxiter => maximum number of iterations before the opitmisation stops,
    disp => to show the results of the optimisation
    '''
    result = scipy.optimize.minimize(optimise_fn, x0, args = (c_ddr, df, f_avg),
            method='SLSQP', bounds = [(0, np.inf) for _ in x0], options = {"maxiter" : 1000, "disp": True})
    x_val.append(result['x'])
    obj_fn.append(result['fun'])

# select the fitted values with the least error
x_sel = x_val[np.argmin(obj_fn)]

# now run the optimise function but in the objective subroutine to return 
# c_w and c_ds
c_w, s = objective(x_sel, c_ddr, df, f_avg)

# Plot
fig, ax = plt.subplots(1,2)
ax[0].scatter(df.index, df['waste'], label = 'expt. waste')
c_w.plot(ax = ax[0], c = 'b')
ax[1].scatter(df.index, df['downstream sorbents'], c = 'k', label = 'expt. sorbents')
ax[1].plot(xnew, s, c= 'k', label = 'downstream sorbent')

ax[0].legend()
plt.legend()

# print the best fitting values
print(x_sel)






