import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, interpolate



def objective(x0, c_ddr, df, f_avg):
    V_waste = 0
    c_w = pd.DataFrame({'waste': [0]*181}) 
    s = [0]
    k_Th = x0[0]
    q_e = x0[1]
    glucose_mass = np.zeros(181)

    for t in range(1, 181):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t)))     
        glucose_mass[t] = (c_ddr - c_ds) * f_avg * 1 
        s.append(c_ds)    
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
    return c_w, s, glucose_mass



def optimise_fn(x0, c_ddr, df, f_avg):
    # print(x0)
    V_waste = 0 #ml
    c_w = pd.DataFrame({'waste': [0]*181}) #mg/ml
    s = pd.DataFrame({'sorbents': [0]*181}) #mg/ml
    k_Th = x0[0] #ml/min/mg
    q_e = x0[1] #mg
    
    for t in range(1, 181):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t))) 
        s.loc[t] = c_ds
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
    return sum(np.sqrt((df['waste']-c_w.loc[df.index, 'waste'])**2))#+ np.sqrt((df['downstream sorbents']-s.loc[df.index, 'sorbents'])**2))

data = {'time':[0, 10, 20, 30, 60, 90, 120, 180], 'waste': [0, 1.9, 11.7, 27.1, 55.0, 76.6, 86.0, 89.5], 
        'downstream sorbents': [0, 22.6, 56.5, 82.7, 101.9, 107.8, 109.2, 108.9]}
df = pd.DataFrame(data)
df['waste'] *= (180.156/1000)
df['downstream sorbents'] *= (180.156/1000)
df.set_index('time', inplace=True)
print(df)

t = 0
x_AC = 200 #g 
time = df.index

#interpolate flow velocity
xnew = np.linspace(0, 180, num=181)
f_avg = np.average([84, 74, 74, 75, 78, 74, 74, 68])

#interpolate downstream dialysate reservoir concentration
c_ddr = 110 * 180.156/1000

# initialise two lists to collect the fitted values and the objective function, i.e., the sum of error
x_val = []
obj_fn = []



#Run simulation
for var in range(2):
    x0 = np.random.random(2)
    result = scipy.optimize.minimize(optimise_fn, x0, args = (c_ddr, df, f_avg),
            method='SLSQP', bounds = [(0, np.inf) for _ in x0], options = {"maxiter" : 1000, "disp": True})
    x_val.append(result['x'])
    obj_fn.append(result['fun'])

# select the fitted values with the least error
x_sel = x_val[np.argmin(obj_fn)]
# run the objective subroutine 
c_w, s, glucose_mass = objective(x_sel, c_ddr, df, f_avg)


# Plot
fig, ax = plt.subplots(1,2)
ax[0].scatter(df.index, df['waste'], label = 'expt. waste')
c_w.plot(ax = ax[0], c = 'b')
ax[1].scatter(df.index, df['downstream sorbents'], c = 'k', label = 'expt. sorbents')
ax[1].plot(xnew, s, c= 'k', label = 'downstream sorbent')
ax[0].legend()
plt.legend()



# convert glucose mass from mg to g
glucose_mass = glucose_mass / 1000

#plot glucose mass vs. time
plt.figure()
plt.plot(range(0, 181), glucose_mass) 
plt.xlabel('Time (min)')
plt.ylabel('Glucose Mass (g)')
# plt.show()

print(x_val, obj_fn)