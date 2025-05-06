import pandas as pd
import numpy as np

# load data and put in single dataframe
data_loc = 'data/'

x_data = np.load(data_loc + '0x.npy')
y_data = np.load(data_loc + '0av.npy')
yerr_data = np.load(data_loc + '0e.npy')

df = pd.DataFrame({
    'Release time (us)': x_data,
    'Surv. prob.': y_data,
    'Error surv. prob.': yerr_data
})

# sort by release time (raw data randomized)
df = df.sort_values(by='Release time (us)', ascending=True)

# save result
print(df)
df.to_csv(data_loc + 'sorted_data.csv', index=False)
