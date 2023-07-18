import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, interpolate
import openpyxl



def objective(x0, c_ddr, data_exp, f_avg, substance):
    V_waste = 0 #ml
    c_w = pd.DataFrame({'waste': [0]*181}) #mg/ml
    s = [0] #mg/ml
    k_Th = x0[0] #ml/mg.min
    q_e = x0[1] #mg/g
    solute_mass = np.zeros(181) #mg

    for t in range(1, 181):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t)))
        solute_mass[t] = (c_ddr - c_ds) * f_avg * 1 
        s.append(c_ds) 
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
    return c_w, s, solute_mass

def optimise_fn(x0, c_ddr, data_exp, f_avg, substance):
    # print(x0)
    V_waste = 0 
    c_w = pd.DataFrame({'waste': [0]*181}) 
    s = pd.DataFrame({'sorbents': [0]*181}) 
    k_Th = x0[0] 
    q_e = x0[1] 
    
    for t in range(1, 181):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t))) 
        s.loc[t] = c_ds
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
        
    return sum(np.sqrt((data_exp['Waste'][substance].astype(float)-c_w.loc[data_w.index, 'waste'])**2))#+ np.sqrt((df['downstream sorbents']-s.loc[df.index, 'sorbents'])**2))



file_path = 'Data.xlsx'
sheet_name = ['Experiment 1', 'Experiment 2'] 
data = {}

for sheet_name in sheet_name:
    cols_name = ['Time', 'Bicarbonate', 'Magnesium', 'Sodium', 'Calcium', 'Glucose']
    
    # Downstream sorbent
    cols = list(range(0, 11))
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=cols, nrows=9, header= 1)
    data_ds = df.T.fillna(0).reset_index(drop=True).drop(columns=[0, 2], index=[0, 9, 10])
    data_ds.columns = cols_name
    data_ds['Time'] = data_ds['Time'].str.replace('t=', '')
    
    # Waste
    cols = list(range(12, 22))
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=cols, nrows=9, header= 0)
    data_w = df.T.fillna(0).reset_index(drop=True).drop(columns= [0, 1, 3], index=[0, 9])
    data_w.columns = cols_name
    data_w['Time'] = data_w['Time'].str.replace('t=', '')
    
    data[sheet_name] = {'Downstream sorbent': data_ds, 'Waste': data_w}

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

t = 0
x_AC = 200 #g 

simulations = []

for sheet_name, data_dict in data.items():
    
    for substance in cols_name[1:]: # it will automatically rotate through all substances for one sheet_name
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_ds[substance].iloc[0]
        data_ds.iloc[0] = 0
        data_w.iloc[0] = 0
        time = data_ds['Time']
        
        #read flow rate directly from the file
        def read_excel_cell(file_path, sheet_name, cell_address):
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            sheet = workbook[sheet_name]
            cell_value = sheet[cell_address].value
            workbook.close()
            return cell_value
        cell_address = 'K20'
        value = read_excel_cell(file_path, sheet_name, cell_address)
        f_avg = float(value)
        
        # initialise two lists to collect the fitted values and the objective function
        x_val = []
        obj_fn = []
        for var in range(1):
            x0 = np.random.random(2)
            # sending back data[sheet_name] to the function instead pf df
            result = scipy.optimize.minimize(optimise_fn, x0, args = (c_ddr, data[sheet_name], f_avg, substance),
                                            method='SLSQP', bounds = [(0, np.inf) for _ in x0], options = {"maxiter" : 1000, "disp": True})
            x_val.append(result['x'])
            obj_fn.append(result['fun'])
        x_sel = x_val[np.argmin(obj_fn)]
        c_w, s, solute_mass = objective(x_sel, c_ddr, df, f_avg, substance)
        
        # print(f"Sheet: {sheet_name}, Solute:{substance}")
        # print(f"Kth:{x_sel[0]} and q_e:{x_sel[1]}")
        
        expt = pd.DataFrame({
            'Solute': [substance],
            'k_Th': [x_sel[0]],
            'q_e': [x_sel[1]]
            })
        
        simulations.append(expt)

# final_df = pd.concat(simulations, ignore_index=True)
# final_df.to_csv('output.csv', index=False)



#Plot
fig, axs = plt.subplots(5, 2, figsize=(12, 15))

# Loop through each solute and plot the data
for sheet_name in sheet_name:
    
    for i, substance in enumerate(cols_name[1:], 0):
        axs[i, 0].scatter(time, data_w[substance], label='expt. waste', c='b')
        c_w.plot(ax = axs[i, 0], c = 'b')
        axs[i, 0].legend()
        
        axs[i, 1].scatter(time, data_ds[substance], c='k', label='expt. sorbents')
        xnew = np.linspace(0, 180, num=181)
        axs[i, 1].plot(xnew, s, c='k', label='downstream sorbent')
        axs[i, 1].legend()
    
    plt.tight_layout()
    plt.show()