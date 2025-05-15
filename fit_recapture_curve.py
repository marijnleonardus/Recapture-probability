# Author: Marijn Venderbosch 
# Date: May 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, proton_mass

# User-defined modules
from modules.classes import BoundStateBasis
from modules.functions import compute_r_squared, load_exp_data, compute_recap_curves, prepare_grids
from modules.units import us, kHz, uK, um

# System Parameters 
mass = 85*proton_mass  # atom mass [kg]
trap_depth = 200*uK  # trap depth [K]
trap_frequency = 54*kHz  # trap frequency [Hz]

# Simulation parameters
nr_temperatures = 10  # number of temperatures to simulate
temperatures = np.linspace(2, 5, nr_temperatures)*uK  # [K]
max_sim_time = 60*us  # [s] maximum simulation time
sim_time_steps = 41
nx = 2048  # number of position points
x_max = 6*um   # half-width [m]

# Raw data 
data_name = 'sorted_data.csv'
unity_surv_time = 5  # [us] where exp data is not flat anymore, to account imaging losses

# (release time in us, survival probability, error in survival probability)
exp_data_x, exp_data_y, exp_data_yerr = load_exp_data('data/', data_name, unity_surv_time)
t_max = max(exp_data_x) #60*us  # [s]
t_steps = len(exp_data_x)  #  20 #number of time steps

# Derived quantities
omega = 2*pi*trap_frequency    # trap angular frequency [rad/s]

# compute time, space and momentum arrays
t_grid, x_grid, dx, k_grid = prepare_grids(t_max, t_steps, x_max, nx)

# prepare wavefunction basis
basis_x, basis_k, wf_energies = BoundStateBasis(omega, mass, trap_depth, x_grid).prepare()

# compute thermally averaged recapture probability curves
thermal_avg_curve, psi_x_evolved = compute_recap_curves(mass, t_grid, basis_k, basis_x, k_grid, dx, wf_energies, temperatures)

# compute R^2 for all simulated temperatures
r_squared_list = []
for curve, T in zip(thermal_avg_curve, temperatures):
    rsquared = compute_r_squared(exp_data_y, curve)
    r_squared_list.append(rsquared)
r_squared_array = np.array(r_squared_list)

# fit 3rd degree plynoial to the R^2 data
coeffs = np.polyfit(temperatures/uK, r_squared_array, deg=3)
plot_x_scale = np.linspace(min(temperatures/uK), max(temperatures/uK), 100)
fitted_curve = np.polyval(coeffs, plot_x_scale)

# obtain best T 
best_fit_temp = plot_x_scale[np.argmax(fitted_curve)]*uK

# plot R^2 vs temperature
fig1, ax1 = plt.subplots()
ax1.scatter(temperatures/uK, r_squared_array, marker='o', color='navy')
ax1.set_xlabel('Temperature [μK]')
ax1.set_ylabel('R^2')
ax1.plot(plot_x_scale, fitted_curve, color='orange', label='Fit')

# plot experimental recapture data along with best temperature fit curve
idx = np.argmin(np.abs(temperatures - best_fit_temp))
curve = thermal_avg_curve[idx]
fig2, ax2 = plt.subplots()
ax2.plot(t_grid/us, curve, label=f'{best_fit_temp/uK:.2f} μK')
ax2.errorbar(exp_data_x/us, exp_data_y, yerr=exp_data_yerr, fmt='o', capsize=5, label='Exp. data')
ax2.set_xlabel('Release time [μs]')
ax2.set_ylabel('Recapture probability')
ax2.set_xlim(0, t_grid.max()/us)
ax2.set_ylim(0, 1.05)
ax2.legend()

plt.savefig('output/fit_exp_data.pdf', dpi=300, bbox_inches='tight')

plt.show()
