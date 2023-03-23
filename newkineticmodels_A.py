import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Flow rate = 20ml/min
df_20 = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
df_20_CtbyC0 = df_20.iloc[:,1]
df_20_Time = df_20.iloc[:,0]

x = 20 #constant inlet concentration C0
df_20_Ct = df_20_CtbyC0 * x

#Kinetic models for Q = 20ml/min
df_Thomas_20 = np.log((x / df_20_Ct)-1)
df_AdamsBohart_20 = np.log(df_20_Ct / x)
df_YoonNelson_20 = np.log(df_20_Ct/(x-df_20_Ct))

plt.plot(df_20_Time, df_Thomas_20, label='Thomas')
plt.plot(df_20_Time, df_AdamsBohart_20, label='Adams-Bohart')
plt.plot(df_20_Time, df_YoonNelson_20, label='Yoon-Nelson')
plt.legend(loc='upper left')

plt.show()