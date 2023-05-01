import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd
from tabulate import tabulate

#Flow rate = 20ml/min
df = pd.read_csv('Biswas_data_flowrate20.csv', header = None)
df_CtbyC0 = df.iloc[:,1]
df_Time = df.iloc[:,0]
df_Ct = df_CtbyC0 * 20

#Calculate q_e i.e. adsorption capacity found from experiment 
Q = 20 
t = np.array(df_Time)
C0 = 20
Ct_e = np.array(df_Ct)

#1
# def calculate_q_total(Q, C0, Ct_e, t):
#     q_total_e = 0
#     C_prev = C0
#     for i in range(len(t)-1):
#         delta_t = t[i+1] - t[i]
#         q_total_e += (C_prev - Ct_e[i]) * delta_t
#         C_prev = Ct_e[i]
#     q_total_e *= Q/1000
#     return q_total_e
# q_total_e = calculate_q_total(Q, C0, Ct_e, t) 

#2
q_total_e = Q/1000 * sum([(C0 - Ct_e[i])* (t[i+1] - t[i]) for i in range(len(t)-1)])

#3 trapezoidal rule 
# q_total_e = Q/1000 * sum([(C0 - (Ct_e[i] + Ct_e[i+1])/2) * (t[i+1] - t[i]) for i in range(len(t)-1)])

#4 
# total = 1000
# q_total_e = 0
# for i in range(len(t)-1):
#     Cavg = 0.5 * (Ct_e[i] + Ct_e[i+1])
#     dt = t[i+1] - t[i]
#     q_total_e += (C0 - Cavg) * dt
# q_total_e *= Q / total

q_e = q_total_e / 2.501

#Thomas model 
k_Th = 0.000516426 
q_0 = 38894.6 
x = 2.501 
Q = 20 
C_0 = 20 
t = df_Time

Th_CtbyC0 = 1 / (1 + np.exp (k_Th * q_0 * x / Q - k_Th * C_0 * t ))
df_Ct_Th = Th_CtbyC0 * 20

# #Calculate q_cal i.e. adsorption capacity calculated from Thomas
Q = 20 
t = np.array(df_Time)
C0 = 20
Ct_Th = np.array(df_Ct_Th)

# #1
def calculate_q_total(Q, C0, Ct_cal, t):
    q_total_cal = 0
    C_prev_cal = C0
    for i in range(len(t)-1):
        delta_t = t[i+1] - t[i]
        q_total_cal += (C_prev_cal - Ct_cal[i]) * delta_t
        C_prev = Ct_e[i]
    q_total_cal *= Q/1000
    return q_total_cal
q_total_cal = calculate_q_total(Q, C0, Ct_cal, t) 

#2 
q_total_Th = Q/1000 * sum([(C0 - Ct_Th[i])* (t[i+1] - t[i]) for i in range(len(t)-1)])

#3 trapezoidal rule 
q_total_cal = Q/1000 * sum([(C0 - (Ct_cal[i] + Ct_cal[i+1])/2) * (t[i+1] - t[i]) for i in range(len(t)-1)])

#4 
total = 1000
q_total_cal = 0
for i in range(len(t)-1):
    Cavg_cal = 0.5 * (Ct_cal[i] + Ct_cal[i+1])
    dt = t[i+1] - t[i]
    q_total_cal += (C0 - Cavg_cal) * dt
q_total_cal *= Q / total

q_cal_Th = q_total_Th / 2.501 

#The sum of the squared of the error (ERRSQ) 
p = 2
sum_squared_errors = 0
for i in range(1, p+1):
        error = q_e - q_cal_Th
        squared_error = error ** 2
        sum_squared_errors += squared_error
ERRSQ_Th = sum_squared_errors

#Hybrid fractional error function (HYBRID)
n = len(df_Ct)
def calculate_hfe(p, n, q_e, q_cal_Th):
    sum_squared_errors = 0
    for i in range(1, n+1):
        error = (q_e - q_cal_Th) / q_e
        squared_error = error ** 2
        sum_squared_errors += squared_error
    hfe_Th = 100 / (n - p) * sum_squared_errors
    return hfe_Th

hfe_Th = calculate_hfe(p, n, q_e, q_cal_Th)

#Marquardt’s percent standard deviation (MPSD)
mpsd_Th = calculate_hfe(p, n, q_e, q_cal_Th)
def calculate_mpsd(p, q_e, q_cal_Th):
    sum_squared_errors = 0
    for i in range(1, n+1):
        error = (q_e - q_cal_Th) / q_e
        squared_error = error ** 2
        sum_squared_errors += squared_error
    mpsd_Th = sum_squared_errors
    return mpsd_Th

#Adams-Bohart 
k_AB = 0.000124468
C_0 = 20
t = df_Time
N_0 = 11187.9
Z = 5 
F = 3.92

AB_CtbyC0 = np.exp(k_AB * C_0 * t - k_AB * N_0 * Z / F)
df_Ct_AB = AB_CtbyC0 * 20 

# #Calculate q_cal i.e. adsorption capacity calculated from Adams-Bohart
Q = 20 
t = np.array(df_Time)
C0 = 20
Ct_AB = np.array(df_Ct_AB)

q_total_AB = Q/1000 * sum([(C0 - Ct_AB[i])* (t[i+1] - t[i]) for i in range(len(t)-1)])

q_cal_AB = q_total_AB / 2.501 

#ERRSQ
p = 2
sum_squared_errors = 0
for i in range(1, p+1):
        error = q_e - q_cal_AB
        squared_error = error ** 2
        sum_squared_errors += squared_error
ERRSQ_AB = sum_squared_errors

#Hybrid fractional error function (HYBRID)
n = len(df_Ct)
def calculate_hfe(p, n, q_e, q_cal_AB):
    sum_squared_errors = 0
    for i in range(1, p+1):
        error = (q_e - q_cal_AB) / q_e
        squared_error = error ** 2
        sum_squared_errors += squared_error
    hfe_AB = 100 / (n - p) * sum_squared_errors
    return hfe_AB
hfe_AB = calculate_hfe(p, n, q_e, q_cal_AB)

#Marquardt’s percent standard deviation (MPSD)
mpsd_AB = calculate_mpsd(p, q_e, q_cal_AB)
def calculate_mpsd(p, q_e, q_cal_AB):
    sum_squared_errors = 0
    for i in range(1, p+1):
        error = (q_e - q_cal_AB) / q_e
        squared_error = error ** 2
        sum_squared_errors += squared_error
    mpsd_AB = sum_squared_errors
    return mpsd_AB

#Yoon-Nelson
k_YN = 0.0103285
torque = 243.189
t = df_Time

YN_CtbyC0 = 1 / (1 + np.exp(k_YN * (torque - t)))
df_Ct_YN = YN_CtbyC0 * 20

#Calculate q_cal i.e. adsorption capacity calculated from Yoon-Nelson
Q = 20 
t = np.array(df_Time)
C0 = 20
Ct_YN = np.array(df_Ct_YN)

q_total_YN = Q/1000 * sum([(C0 - Ct_YN[i])* (t[i+1] - t[i]) for i in range(len(t)-1)])
q_cal_YN = q_total_YN / 2.501 

#ERRSQ 
p = 2
sum_squared_errors = 0
for i in range(1, p+1):
        error = q_e - q_cal_YN
        squared_error = error ** 2
        sum_squared_errors += squared_error
ERRSQ_YN = sum_squared_errors

#Hybrid fractional error function (HYBRID)
n = len(df_Ct)
def calculate_hfe(p, n, q_e, q_cal_YN):
    sum_squared_errors = 0
    for i in range(1, n+1):
        error = (q_e - q_cal_YN) / q_e
        squared_error = error ** 2
        sum_squared_errors += squared_error
    hfe_YN = 100 / (n - p) * sum_squared_errors
    return hfe_YN

hfe_YN = calculate_hfe(p, n, q_e, q_cal_YN)

#Marquardt’s percent standard deviation (MPSD)
mpsd_YN = calculate_mpsd(p, q_e, q_cal_YN)
def calculate_mpsd(p, q_e, q_cal_YN):
    sum_squared_errors = 0
    for i in range(1, p+1):
        error = (q_e - q_cal_YN) / q_e
        squared_error = error ** 2
        sum_squared_errors += squared_error
    mpsd_YN = sum_squared_errors
    return mpsd_YN

data = [['Thomas', ERRSQ_Th, hfe_Th, mpsd_Th], \
       ['Adams-Bohart', ERRSQ_AB, hfe_AB, mpsd_AB], \
       ['Yoon-Nelson', ERRSQ_YN, hfe_YN, mpsd_YN]]

headers=["Model","ERRSQ", "HFE", "MPSD"]
print (tabulate(data, headers))