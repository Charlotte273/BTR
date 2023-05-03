import numpy as np 
import pandas as pd
from tabulate import tabulate

#Flow rate = 20ml/min
#PART 1: EXPERIMENTAL Ct/C0

df = pd.read_csv('Biswas_data_flowrate20.csv', header = None)
df_CtbyC0 = df.iloc[:,1]
df_20_Time = df.iloc[:,0]

#PART 2: PREDICTED Ct/C0

#Thomas model 
k_Th = 0.000516426 
q_e = 38894.6
x = 2.501 
Q = 20 
C_0 = 20 
t = df_20_Time
Th_CtbyC0 = 1 / (1 + np.exp (k_Th * q_e * x / Q - k_Th * C_0 * t ))

#Adams-Bohart 
k_AB = 0.000124468
C_0 = 20
t = df_20_Time
N_0 = 11187.9
Z = 5 
F = 3.92
AB_CtbyC0 = np.exp(k_AB * C_0 * t - k_AB * N_0 * Z / F)

#Yoon-Nelson
k_YN = 0.0103285
torque = 243.189
t = df_20_Time
YN_CtbyC0 = 1 / (1 + np.exp(k_YN * (torque - t)))

#ERROR ANALYSIS USING Ct/C0

#Adams-Bohart

#The sum of the squared of the error (ERRSQ) 
n = len(df_20_Time)
sum_squared_errors = 0
for i in range(n):
        error = df_CtbyC0[i] - AB_CtbyC0[i]
        squared_error = error ** 2
        sum_squared_errors += np.sum(squared_error)
ERRSQ_AB = sum_squared_errors

#Hybrid fractional error function (HYBRID)
n = len(df_20_Time)
p = 2
def calculate_hfe(n, p, df_CtbyC0, AB_CtbyC0):
        sum_squared_errors = sum(
            ((df_CtbyC0[i] - AB_CtbyC0[i]) **2 / df_CtbyC0[i]) for i in range(n))
        return 100 / (n - p) * sum_squared_errors
hfe_AB = calculate_hfe(n, p, df_CtbyC0, AB_CtbyC0)

#Marquardt’s percent standard deviation (MPSD)
n = len(df_20_Time)
p = 2
def calculate_mpsd(n, p, df_CtbyC0, AB_CtbyC0):
        # return sum(((df_CtbyC0[i] - AB_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n))
        return 100 * np.sqrt ((1/ (n - p)) * sum(((df_CtbyC0[i] - AB_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n)))
mpsd_AB = calculate_mpsd(n, p, df_CtbyC0, AB_CtbyC0)
print(ERRSQ_AB, hfe_AB, mpsd_AB)


#Thomas 

#The sum of the squared of the error (ERRSQ) 
n = len(df_20_Time)
sum_squared_errors = 0
for i in range(n):
        error = df_CtbyC0[i] - Th_CtbyC0[i]
        squared_error = error ** 2
        sum_squared_errors += np.sum(squared_error)
ERRSQ_Th = sum_squared_errors

#Hybrid fractional error function (HYBRID)
n = len(df_20_Time)
p = 2
def calculate_hfe(n, p, df_CtbyC0, Th_CtbyC0):
        sum_squared_errors = sum(
            ((df_CtbyC0[i] - Th_CtbyC0[i])**2 / df_CtbyC0[i]) for i in range(n))
        return 100 / (n - p) * sum_squared_errors
hfe_Th = calculate_hfe(n, p, df_CtbyC0, Th_CtbyC0)

#Marquardt’s percent standard deviation (MPSD)
def calculate_mpsd(n, p, df_CtbyC0, Th_CtbyC0):
        # return sum(((df_CtbyC0[i] - Th_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n))
        return 100 * np.sqrt ((1/ (n - p)) * sum(((df_CtbyC0[i] - Th_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n)))
mpsd_Th = calculate_mpsd(n, p, df_CtbyC0, Th_CtbyC0)

print(ERRSQ_Th, hfe_Th, mpsd_Th)

#Yoon-Nelson
#The sum of the squared of the error (ERRSQ) 
n = len(df_20_Time)
sum_squared_errors = 0
for i in range(n):
        error = df_CtbyC0[i] - YN_CtbyC0[i]
        squared_error = error ** 2
        sum_squared_errors += np.sum(squared_error)
ERRSQ_YN = sum_squared_errors

#Hybrid fractional error function (HYBRID)
n = len(df_20_Time)
p = 2
def calculate_hfe(n, p, df_CtbyC0, YN_CtbyC0):
        sum_squared_errors = sum(
            ((df_CtbyC0[i] - YN_CtbyC0[i])**2 / df_CtbyC0[i])
            for i in range(n))
        return 100 / (n - p) * sum_squared_errors

hfe_YN = calculate_hfe(n, p, df_CtbyC0, YN_CtbyC0)

#Marquardt’s percent standard deviation (MPSD)
def calculate_mpsd(n, p, df_CtbyC0, YN_CtbyC0):
        # return sum(((df_CtbyC0[i] - YN_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n))
        return 100 * np.sqrt ((1/ (n - p)) * sum(((df_CtbyC0[i] - YN_CtbyC0[i]) / df_CtbyC0[i]) **2 for i in range(n)))
mpsd_YN = calculate_mpsd(n, p, df_CtbyC0, YN_CtbyC0)

print(ERRSQ_YN, hfe_YN, mpsd_YN)