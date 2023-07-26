import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#Analyze how k_Th and q_e change with different flow rate and initial solute conc.
df = pd.read_csv('output_1.csv', delimiter=';')

# Experiments which have different flow rate
experiments = ['Experiment 0', 'Experiment 1', 'Experiment 2']

# Flow rate mapping for each experiment
flow_rate_mapping = {'Experiment 0': 60, 'Experiment 1': 100, 'Experiment 2': 150}

# Get a list of unique solutes in the DataFrame
unique_solutes = df['Solute'].unique()

# Create a figure for k_Th with subplots for each solute
fig_k_Th, axs_k_Th = plt.subplots(1, len(unique_solutes), figsize=(15, 5))

# Create a figure for q_e with subplots for each solute
fig_q_e, axs_q_e = plt.subplots(1, len(unique_solutes), figsize=(15, 5))

# Loop through each solute and plot the corresponding subplot for k_Th and q_e
for i, solute in enumerate(unique_solutes):
    # Filter the DataFrame to get the rows for the desired experiments and current solute
    experiment_solute = df[(df['Sheet_Name'].isin(experiments)) & (df['Solute'] == solute)]
    
    # Get k_Th values for each flow rate (60, 100, 150) using the mapping
    k_Th_values = []
    q_e_values = []
    for experiment in experiments:
        flow_rate = flow_rate_mapping[experiment]
        k_Th = experiment_solute.loc[experiment_solute['Sheet_Name'] == experiment, 'k_Th'].values[0]
        k_Th_values.append((flow_rate, k_Th))
        
        q_e = experiment_solute.loc[experiment_solute['Sheet_Name'] == experiment, 'q_e'].values[0]
        q_e_values.append((flow_rate, q_e))
    
    # Sort the values based on flow rate to ensure the correct order in the plot
    k_Th_values.sort(key=lambda x: x[0])
    q_e_values.sort(key=lambda x: x[0])
    
    # Separate flow rates and k_Th values for the plot
    flow_rates_k_Th, k_Th = zip(*k_Th_values)
    
    # Plot the current solute in the corresponding subplot for k_Th
    axs_k_Th[i].plot(flow_rates_k_Th, k_Th, marker='o', linestyle='-')
    axs_k_Th[i].set_xlabel('Flow Rate')
    axs_k_Th[i].set_ylabel('k_Th')
    axs_k_Th[i].set_title(f'{solute}')
    axs_k_Th[i].grid(True)
    
    # Perform linear regression to get the line equation
    x = np.array(flow_rates_k_Th)
    y = np.array(k_Th)
    
    slope, intercept = np.polyfit(x, y, 1)
    
    # Plot the regression line
    axs_k_Th[i].plot(x, slope * x + intercept, color='red', linestyle='--', label='Regression Line')
    
    # Add the equation to the plot
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    axs_k_Th[i].text(0.1, 0.8, equation, transform=axs_k_Th[i].transAxes, fontsize=12)
    
    axs_k_Th[i].legend()
    
    # Separate flow rates and q_e values for the plot
    flow_rates_q_e, q_e = zip(*q_e_values)
    
    # Plot the current solute in the corresponding subplot for q_e
    axs_q_e[i].plot(flow_rates_q_e, q_e, marker='o', linestyle='-')
    axs_q_e[i].set_xlabel('Flow Rate')
    axs_q_e[i].set_ylabel('q_e')
    axs_q_e[i].set_title(f'{solute}')
    axs_q_e[i].grid(True)
    
    # Perform linear regression to get the line equation
    x = np.array(flow_rates_q_e)
    y = np.array(q_e)
    
    slope, intercept = np.polyfit(x, y, 1)
    
    # Plot the regression line
    axs_q_e[i].plot(x, slope * x + intercept, color='red', linestyle='--', label='Regression Line')
    
    # Add the equation to the plot
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    axs_q_e[i].text(0.1, 0.8, equation, transform=axs_q_e[i].transAxes, fontsize=12)
    
    axs_q_e[i].legend()

# # Adjust the layout of the subplots for better spacing for k_Th figure
# fig_k_Th.tight_layout()

# # Save the figure for k_Th to a file
# fig_k_Th.savefig('Q_k_Th_figures.png')

# # Adjust the layout of the subplots for better spacing for q_e figure
# fig_q_e.tight_layout()

# # Save the figure for q_e to a file
# fig_q_e.savefig('Q_q_e_figures.png')



#Analyze how k_Th and q_e change with different flow rate and initial solute conc.
df = pd.read_csv('output_1.csv', delimiter=';')

# Experiments which have different flow rate
experiments = ['Experiment 3', 'Experiment 1', 'Experiment 5']

# Flow rate mapping for each experiment
initial_conc_mapping = {'Experiment 3': 15, 'Experiment 1': 20, 'Experiment 5': 25}

# Get a list of unique solutes in the DataFrame
unique_solutes = df['Solute'].unique()

# Create a figure for k_Th with subplots for each solute
fig_k_Th, axs_k_Th = plt.subplots(1, len(unique_solutes), figsize=(15, 5))

# Create a figure for q_e with subplots for each solute
fig_q_e, axs_q_e = plt.subplots(1, len(unique_solutes), figsize=(15, 5))

# Loop through each solute and plot the corresponding subplot for k_Th and q_e
for i, solute in enumerate(unique_solutes):
    # Filter the DataFrame to get the rows for the desired experiments and current solute
    experiment_solute = df[(df['Sheet_Name'].isin(experiments)) & (df['Solute'] == solute)]
    
    # Get k_Th values for each solute concentration using the mapping
    k_Th_values = []
    q_e_values = []
    for experiment in experiments:
        initial_conc = initial_conc_mapping[experiment]
        k_Th = experiment_solute.loc[experiment_solute['Sheet_Name'] == experiment, 'k_Th'].values[0]
        k_Th_values.append((initial_conc, k_Th))
        
        q_e = experiment_solute.loc[experiment_solute['Sheet_Name'] == experiment, 'q_e'].values[0]
        q_e_values.append((initial_conc, q_e))
    
    # Sort the values based on flow rate to ensure the correct order in the plot
    k_Th_values.sort(key=lambda x: x[0])
    q_e_values.sort(key=lambda x: x[0])
    
    # Separate flow rates and k_Th values for the plot
    initial_conc_k_Th, k_Th = zip(*k_Th_values)
    
    # Plot the current solute in the corresponding subplot for k_Th
    axs_k_Th[i].plot(initial_conc_k_Th, k_Th, marker='o', linestyle='-')
    axs_k_Th[i].set_xlabel('Initial concentration')
    axs_k_Th[i].set_ylabel('k_Th')
    axs_k_Th[i].set_title(f'{solute}')
    axs_k_Th[i].grid(True)
    
    # Perform linear regression to get the line equation
    x = np.array(initial_conc_k_Th)
    y = np.array(k_Th)
    
    slope, intercept = np.polyfit(x, y, 1)
    
    # Plot the regression line
    axs_k_Th[i].plot(x, slope * x + intercept, color='red', linestyle='--', label='Regression Line')
    
    # Add the equation to the plot
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    axs_k_Th[i].text(0.1, 0.8, equation, transform=axs_k_Th[i].transAxes, fontsize=12)
    
    axs_k_Th[i].legend()
    
    # Separate flow rates and q_e values for the plot
    initial_conc_q_e, q_e = zip(*q_e_values)
    
    # Plot the current solute in the corresponding subplot for q_e
    axs_q_e[i].plot(initial_conc_q_e, q_e, marker='o', linestyle='-')
    axs_q_e[i].set_xlabel('Initial concentration')
    axs_q_e[i].set_ylabel('q_e')
    axs_q_e[i].set_title(f'{solute}')
    axs_q_e[i].grid(True)
    
    # Perform linear regression to get the line equation
    x = np.array(initial_conc_q_e)
    y = np.array(q_e)
    slope, intercept = np.polyfit(x, y, 1)
    
    # Plot the regression line
    axs_q_e[i].plot(x, slope * x + intercept, color='red', linestyle='--', label='Regression Line')
    
    # Add the equation to the plot
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    axs_q_e[i].text(0.1, 0.8, equation, transform=axs_q_e[i].transAxes, fontsize=12)
    
    axs_q_e[i].legend()

# Adjust the layout of the subplots for better spacing for k_Th figure
fig_k_Th.tight_layout()

# Save the figure for k_Th to a file
fig_k_Th.savefig('x0_k_Th_figures.png')

# Adjust the layout of the subplots for better spacing for q_e figure
fig_q_e.tight_layout()

# Save the figure for q_e to a file
fig_q_e.savefig('x_0_q_e_figures.png')

