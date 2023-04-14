
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('exptal_data.csv', header = 1, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])

#GLUCOSE
df_glucose = df.iloc[138:149]
df_Downstream_sorbent9 = df_glucose.iloc[:,8]
df_Donwstream_10L_reservoir9 = df_glucose.iloc[:,7]

df_Ct_9 = df_Downstream_sorbent9.dropna()
df_C0_9 = df_Donwstream_10L_reservoir9.dropna()
df_Ct_9 = df_Ct_9.astype(float)
df_C0_9 = df_C0_9.astype(float)

#Thomas
df_glucose['ln(C0byCt-1)'] = np.log((df_C0_9 / df_Ct_9)-1)
#Adams-Bohart 
df_glucose['ln(CtbyC0)'] = np.log((df_Ct_9 / df_C0_9))
#Yoon-Nelson 
df_glucose['ln(Ctby(C0-Ct))'] = np.log(df_Ct_9/(df_C0_9-df_Ct_9))

#PHOSPHATE
df_phosphate = df.iloc[120:131]
df_Downstream_sorbent8 = df_phosphate.iloc[:,8]
df_Donwstream_10L_reservoir8 = df_phosphate.iloc[:,7]

df_Ct_8 = df_Downstream_sorbent8.dropna()
df_C0_8 = df_Donwstream_10L_reservoir8.dropna()
df_Ct_8 = df_Ct_8.astype(float)
df_C0_8 = df_C0_8.astype(float)

#Thomas
df_phosphate['ln(C0byCt-1)'] = np.log((df_C0_8 / df_Ct_8)-1)
#Adams-Bohart 
df_phosphate['ln(CtbyC0)'] = np.log((df_Ct_8 / df_C0_8))
#Yoon-Nelson 
df_phosphate['ln(Ctby(C0-Ct))'] = np.log(df_Ct_8/(df_C0_9-df_Ct_9))

#plotting
fig, ax = plt.subplots(2,3, figsize = (18,18))
ax[0,0].plot(df_glucose['T='], df_glucose['ln(C0byCt-1)'])
ax[0,0].set_title("Glucose_Thomas")

ax[0,1].plot(df_glucose['T='], df_glucose['ln(CtbyC0)'])
ax[0,1].set_title("Glucose_Adams-Bohart")

ax[0,2].plot(df_glucose['T='], df_glucose['ln(Ctby(C0-Ct))'])
ax[0,2].set_title("Glucose_Yoon-Nelson")

ax[1,0].plot(df_phosphate['T='], df_phosphate['ln(C0byCt-1)'])
ax[1,0].set_title("Phosphate_Thomas")

ax[1,1].plot(df_phosphate['T='], df_phosphate['ln(CtbyC0)'])
ax[1,1].set_title("Phosphate_Adams-Bohart")

ax[1,2].plot(df_phosphate['T='], df_phosphate['ln(Ctby(C0-Ct))'])
ax[1,2].set_title("Phosphate_Yoon-Nelson")

plt.show()
