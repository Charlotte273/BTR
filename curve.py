import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('exptal_data.csv', header = 1, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])

#Solute 1: Sodium
df_sodium = df.iloc[1:12]
df_Downstream_sorbents1 = df_sodium.iloc[:,8]
df_Donwstream_10L_reservoir1 = df_sodium.iloc[:,7]

df_Ct_1 = df_Downstream_sorbents1.dropna()
df_C0_1 = df_Donwstream_10L_reservoir1.dropna()

df_Ct_1 = df_Ct_1.astype(int)
df_C0_1 = df_C0_1.astype(int)
df_sodium['CtbyC0'] =  df_Ct_1 / df_C0_1

#Solute 2: Chloride
df_chloride = df.iloc[18:29]
df_Downstream_sorbents2 = df_chloride.iloc[:,8]
df_Donwstream_10L_reservoir2 = df_chloride.iloc[:,7]

df_Ct_2 = df_Downstream_sorbents2.dropna()
df_C0_2 = df_Donwstream_10L_reservoir2.dropna()

df_Ct_2 = df_Ct_2.astype(int)
df_C0_2 = df_C0_2.astype(int)
df_chloride['CtbyC0'] =  df_Ct_2 / df_C0_2

#Solute 3: Potassium 
df_potassium = df.iloc[35:46]
df_Downstream_sorbents3 = df_potassium.iloc[:,8]
df_Donwstream_10L_reservoir3 = df_potassium.iloc[:,7]

df_Ct_3 = df_Downstream_sorbents3.dropna()
df_C0_3 = df_Donwstream_10L_reservoir3.dropna()

df_Ct_3 = df_Ct_3.astype(float)
df_C0_3 = df_C0_3.astype(float)
df_potassium['CtbyC0'] =  df_Ct_3 / df_C0_3

#Solute 4: Bicarbonate
df_bicarbonate = df.iloc[52:63]
df_Downstream_sorbents4 = df_bicarbonate.iloc[:,8]
df_Donwstream_10L_reservoir4 = df_bicarbonate.iloc[:,7]

df_Ct_4 = df_Downstream_sorbents4.dropna()
df_C0_4 = df_Donwstream_10L_reservoir4.dropna()

df_Ct_4 = df_Ct_4.astype(float)
df_C0_4 = df_C0_4.astype(float)
df_bicarbonate['CtbyC0'] =  df_Ct_4 / df_C0_4

#Solute 5: Lactate
df_lactate = df.iloc[69:80]
df_Downstream_sorbents5 = df_lactate.iloc[:,8]
df_Donwstream_10L_reservoir5 = df_lactate.iloc[:,7]

df_Ct_5 = df_Downstream_sorbents5.dropna()
df_C0_5 = df_Donwstream_10L_reservoir5.dropna()

df_Ct_5 = df_Ct_5.astype(float)
df_C0_5 = df_C0_5.astype(float)
df_lactate['CtbyC0'] =  df_Ct_5 / df_C0_5

#Solute 6: Calcium
df_calcium = df.iloc[86:97]
df_Downstream_sorbents6 = df_calcium.iloc[:,8]
df_Donwstream_10L_reservoir6 = df_calcium.iloc[:,7]

df_Ct_6 = df_Downstream_sorbents6.dropna()
df_C0_6 = df_Donwstream_10L_reservoir6.dropna()

df_Ct_6 = df_Ct_6.astype(float)
df_C0_6 = df_C0_6.astype(float)
df_calcium['CtbyC0'] =  df_Ct_6 / df_C0_6

#Solute 7: Magnesium
df_magnesium = df.iloc[103:114]
df_Downstream_sorbents7 = df_magnesium.iloc[:,8]
df_Donwstream_10L_reservoir7 = df_magnesium.iloc[:,7]

df_Ct_7 = df_Downstream_sorbents7.dropna()
df_C0_7 = df_Donwstream_10L_reservoir7.dropna()

df_Ct_7 = df_Ct_7.astype(float)
df_C0_7 = df_C0_7.astype(float)
df_magnesium['CtbyC0'] =  df_Ct_7 / df_C0_7

#Solute 8: Phosphate
df_phosphate = df.iloc[120:131]
df_Downstream_sorbent8 = df_phosphate.iloc[:,8]
df_Donwstream_10L_reservoir8 = df_phosphate.iloc[:,7]

df_Ct_8 = df_Downstream_sorbent8.dropna()
df_C0_8 = df_Donwstream_10L_reservoir8.dropna()

df_Ct_8 = df_Ct_8.astype(float)
df_C0_8 = df_C0_8.astype(float)
df_phosphate['CtbyC0'] =  df_Ct_8 / df_C0_8

#Solute 9: Glucose
df_glucose = df.iloc[138:149]

df_Downstream_sorbent9 = df_glucose.iloc[:,8]
df_Donwstream_10L_reservoir9 = df_glucose.iloc[:,7]

df_Ct_9 = df_Downstream_sorbent9.dropna()
df_C0_9 = df_Donwstream_10L_reservoir9.dropna()

df_Ct_9 = df_Ct_9.astype(float)
df_C0_9 = df_C0_9.astype(float)
df_glucose['CtbyC0'] =  df_Ct_9 / df_C0_9

#plot the breakthrough curves
fig, ax = plt.subplots(3,3, figsize = (18,18))
ax[0,0].plot(df_sodium['T='], df_sodium['CtbyC0'] )
ax[0,0].set_title("Sodium")

ax[0,1].plot(df_chloride['T='], df_chloride['CtbyC0'])
ax[0,1].set_title("Chloride")

ax[0,2].plot(df_potassium['T='], df_potassium['CtbyC0'])
ax[0,2].set_title("Potassium")

ax[1,0].plot(df_bicarbonate['T='], df_bicarbonate['CtbyC0'])
ax[1,0].set_title("Bicarbonate")

ax[1,1].plot(df_lactate['T='], df_lactate['CtbyC0'])
ax[1,1].set_title("Lactate")

ax[1,2].plot(df_calcium['T='], df_calcium['CtbyC0'])
ax[1,2].set_title("Calcium")

ax[2,0].plot(df_magnesium['T='], df_magnesium['CtbyC0'])
ax[2,0].set_title("Magnesium")

ax[2,1].plot(df_phosphate['T='], df_phosphate['CtbyC0'])
ax[2,1].set_title("Phosphate")

ax[2,2].plot(df_glucose['T='], df_glucose['CtbyC0'])
ax[2,2].set_title("Glucose")

fig.supxlabel('Time')
fig.supylabel('Ct/C0')

font1 = {'family':'serif','color':'blue','size':30}
fig.suptitle('Breakthrough Curve', fontdict = font1)
plt.show()