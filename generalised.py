# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:33:38 2023

@author: P70073624
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, interpolate
import openpyxl



def objective(x0, c_ddr, data_ds, data_w, f_avg, substance):
    V_waste = 0 #ml
    c_w = pd.DataFrame({'waste': [0]*t_end}) #mg/ml
    s = [0] #mg/ml
    k_Th = x0[0] #ml/mg.min
    q_e = x0[1] #mg/g
    solute_mass = np.zeros(t_end) #mg

    for t in range(1, t_end):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t)))
        solute_mass[t] = (c_ddr - c_ds) * f_avg * 1 
        s.append(c_ds) 
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
    return c_w, s, solute_mass

def optimise_fn(x0, c_ddr, data_ds, data_w, f_avg, substance):
    # print(x0)
    V_waste = 0 
    c_w = pd.DataFrame({'waste': [0]*t_end}) 
    s = pd.DataFrame({'sorbents': [0]*t_end}) 
    k_Th = x0[0] 
    q_e = x0[1] 
    
    for t in range(1, t_end):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t))) 
        s.loc[t] = c_ds
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
     
    return sum(np.sqrt((data_ds[substance].astype(float)-s.loc[data_ds.index, 'sorbents'])**2)) +sum(np.sqrt((data_w[substance].astype(float)-c_w.loc[data_w.index, 'waste'])**2))

#read flow rate directly from the file
def read_excel_cell(file_path, sheet_name, cell_address):
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook[sheet_name]
    cell_value = sheet[cell_address].value
    workbook.close()
    return cell_value

file_path = 'Data.xlsx'
sheet_name = ['Experiment 3', 'Experiment 2'] 
data = {}

for sheet_name in sheet_name:
    cols_name = ['Time', 'Bicarbonate', 'Magnesium', 'Sodium', 'Calcium', 'Glucose']
    
    # Downstream sorbent
    
    df = pd.read_excel(file_path, sheet_name=sheet_name, header = 0)
    df = df.T
    data_ds = df[df.iloc[:,1] == 'line'].reset_index(drop = True).dropna(axis = 1, how = 'all').drop(columns=[0, 1, 3])
    data_ds.columns = cols_name
    data_ds['Time'] = data_ds['Time'].str.replace('t=', '').astype(int)
    data_ds.set_index(['Time'], inplace = True)
    
    # Waste
    df = pd.read_excel(file_path, sheet_name=sheet_name, header= 0)
    df = df.T
    data_w = df[df.iloc[:,1] == 'reservoir'].reset_index(drop = True).dropna(axis = 1, how = 'all').drop(columns=[0, 1, 3])
    data_w.columns = cols_name
    data_w['Time'] = data_w['Time'].str.replace('t=', '').astype(int) 
    data_w.set_index(['Time'], inplace = True)
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    f_avg = df[df.iloc[:,0] == 'average flow rate'].iloc[0,1]
    # cell_address = 'K22' # SHIFT this to a better location.
    # value = read_excel_cell(file_path, sheet_name, cell_address)
    # f_avg = float(value)
    
    c_ddr = data_ds.iloc[0].copy(deep = True)
    
    data_ds.iloc[0] = 0
    data_w.iloc[0] = 0
    # print(c_ddr)
    data[sheet_name] = {'Downstream sorbent': data_ds, 'Waste': data_w, 'Flow rate':f_avg, 'Initial conc': c_ddr}

    # print(f"Sheet: {sheet_name}")
    # print("Downstream sorbent concentration:")
    # print(data_ds)
    # print("Waste concentration:")
    # print(data_w)

molecular_weight = {'Bicarbonate': 61.0168, 
                    'Magnesium': 24.3050, 
                    'Sodium': 22.9898, 
                    'Calcium': 40.0784, 
                    'Glucose': 180.156
                    }

 

x_AC = 200 #g

for sheet_name, data_dict in data.items():    
    
    simulations = pd.DataFrame(columns = ['Solute', 'k_Th', 'q_e'])
    for substance in cols_name[1:]: # it will automatically rotate through all substances for one sheet_name
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_dict['Initial conc'][substance]* molecular_weight[substance] / 1000
        time = data_ds.index
        f_avg = data_dict['Flow rate']
        t_end = data_ds.index[-1]+1
        
        
        # initialise two lists to collect the fitted values and the objective function
        x_val = []
        obj_fn = []
        for var in range(10):
            x0 = np.random.random(2)
            # sending back data[sheet_name] to the function instead pf df
            result = scipy.optimize.minimize(optimise_fn, x0, args = (c_ddr, data_ds, data_w, f_avg, substance),
                                            method='SLSQP', bounds = [(0, np.inf) for _ in x0], options = {"maxiter" : 1000, "disp": False})
            x_val.append(result['x'])
            obj_fn.append(result['fun'])
        x_sel = x_val[np.argmin(obj_fn)]
        
        
        # print(f"Sheet: {sheet_name}, Solute:{substance}")
        # print(f"Kth:{x_sel[0]} and q_e:{x_sel[1]}")
        
        expt = pd.DataFrame({
            'Solute': [substance],
            'k_Th': [x_sel[0]],
            'q_e': [x_sel[1]]
            })
        
        simulations=pd.concat([simulations,expt])

# final_df = pd.concat(simulations, ignore_index=True)
# final_df.to_csv('output.csv', index=False)

    simulations.set_index(['Solute'], inplace = True)
#%%
#Plot
    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    
    for i, substance in enumerate(cols_name[1:], 0):
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_dict['Initial conc'][substance] * molecular_weight[substance] / 1000
        t_end = data_ds.index[-1]+1
        time = data_ds.index
        x_sel = simulations.loc[substance]
        c_w, s, solute_mass = objective(x_sel, c_ddr, data_ds, data_w, f_avg, substance)
        axs[i, 0].scatter(time, data_w[substance], label='expt. waste', c='b')
        c_w.plot(ax = axs[i, 0], c = 'b')
        axs[i, 0].legend()
        
        axs[i, 1].scatter(time, data_ds[substance], c='k', label='expt. sorbents')
        axs[i, 1].plot(range(0,t_end), s, c='k', label='downstream sorbent')
        axs[i, 1].legend()
    
    plt.tight_layout()
    plt.show()