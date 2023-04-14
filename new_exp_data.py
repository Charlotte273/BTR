import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Flow rate = 20ml/min
df_20 = pd.read_csv('Biswas_data_flowrate20.csv', header = 0)
df_20_CtbyC0 = df_20.iloc[:,1]
df_20_Time = df_20.iloc[:,0]

print(df_20_Time)

#Flow rate = 15ml/min
df_15 = pd.read_csv('Biswas_data_flowrate15.csv', header = 0)
df_15_CtbyC0 = df_15.iloc[:,1]
df_15_Time = df_15.iloc[:,0]

#Flow rate = 10ml/min
df_10 = pd.read_csv('Biswas_data_flowrate10.csv', header = 0)
df_10_CtbyC0 = df_10.iloc[:,1]
df_10_Time = df_10.iloc[:,0]

plt.scatter(df_20_Time, df_20_CtbyC0, label='Flow rate 20 ml/min')
plt.scatter(df_15_Time, df_15_CtbyC0, label='Flow rate 15 ml/min')
plt.scatter(df_10_Time, df_10_CtbyC0, label='Flow rate 10 ml/min')
plt.legend(loc="upper left")
plt.xlabel('Time (min)')
plt.ylabel('Ct/C0')
plt.show()
