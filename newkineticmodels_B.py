import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Flow rate = 20ml/min
df_20 = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
df_20_Time = df_20.iloc[:,0]

#Thomas model
x = 20 #C0
k_Th = 6.15 * 10 ** -4
q0 = 57229.79
m = 10
Q = 20
t = df_20_Time
A = (k_Th * q0 * m)/Q - (k_Th * x * t)
y = x / np.float_power(np.exp, A) + 1
print(y)
