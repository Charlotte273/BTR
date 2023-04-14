import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#PART 1: EXPERIMENTAL CURVES

#Flow rate = 20ml/min
df_20 = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
df_20_CtbyC0 = df_20.iloc[:,1]
df_20_Time = df_20.iloc[:,0]

#Flow rate = 15ml/min
df_15 = pd.read_csv('Biswas_data_flowrate15.csv', header = 0)
df_15_CtbyC0 = df_15.iloc[:,1]
df_15_Time = df_15.iloc[:,0]

#Flow rate = 10ml/min
df_10 = pd.read_csv('Biswas_data_flowrate10.csv', header = 0)
df_10_CtbyC0 = df_10.iloc[:,1]
df_10_Time = df_10.iloc[:,0]

plt.scatter(df_20_Time, df_20_CtbyC0, label='Flow rate 20 ml/min')
# plt.scatter(df_15_Time, df_15_CtbyC0, label='Flow rate 15 ml/min')
# plt.scatter(df_10_Time, df_10_CtbyC0, label='Flow rate 10 ml/min')

#PART 2: PREDICTED CURVES FOR Q = 20

#Thomas model 
k_Th = 0.000516426 
q_e = 38894.6
x = 2.501 
Q = 20 
C_0 = 20 
t = df_20_Time

Th_CtbyC0 = 1 / (1 + np.exp (k_Th * q_e * x / Q - k_Th * C_0 * t ))
plt.plot(df_20_Time, Th_CtbyC0, label='Thomas')

#Adams-Bohart 
k_AB = 0.000124468
C_0 = 20
t = df_20_Time
N_0 = 11187.9
Z = 5 
F = 3.92

AD_CtbyC0 = np.exp(k_AB * C_0 * t - k_AB * N_0 * Z / F)
plt.plot(df_20_Time, AD_CtbyC0, label='Adams-Bohart')

#Yoon-Nelson
k_YN = 0.0103285
torque = 243.189
t = df_20_Time

YN_CtbyC0 = 1 / (1 + np.exp(k_YN * (torque - t)))
plt.plot(df_20_Time, YN_CtbyC0, label='Yoon-Nelson')

plt.legend(loc="upper left")
plt.xlabel('Time (min)')
plt.ylabel('Ct/C0')
plt.show()