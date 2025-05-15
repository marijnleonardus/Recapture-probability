# Author: Marijn Venderbosch 
# Date: May 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, proton_mass

# User-defined modules
from modules.classes import BoundStateBasis
from modules.units import us, kHz, uK, um
from modules.functions import prepare_grids, compute_recap_curves

# System Parameters 
mass = 88*proton_mass  # atom mass [kg]
trap_depth = 50*uK  # trap depth [K]
trap_frequency = 25*kHz  # trap frequency [Hz]

# Simulation parameters
nr_temperatures = 10  # number of temperatures to simulate
temperatures = np.array([0.001, 0.73])*uK  # [K]
t_max = 100*us  # [s] maximum simulation time
t_steps = 41
nx = 2048  # number of position points
x_max = 6*um   # half-width [m]
initial_state = 0  # for wave function expansion plot

# Derived quantities
omega = 2*pi*trap_frequency    # trap angular frequency [rad/s]

# compute time, space and momentum arrays
t_grid, x_grid, dx, k_grid = prepare_grids(t_max, t_steps, x_max, nx)

# prepare wavefunction basis
basis_x, basis_k, wf_energies = BoundStateBasis(omega, mass, trap_depth, x_grid).prepare()

# compute thermally averaged recapture probability curves
thermal_avg_curve, psi_x_evolved = compute_recap_curves(mass, t_grid, basis_k, basis_x, k_grid, dx, wf_energies, temperatures)

# plot recapture probability curves
fig, ax = plt.subplots()
for T in np.atleast_1d(temperatures):
    idx = np.argmin(np.abs(temperatures - T))
    curve = thermal_avg_curve[idx]
    ax.plot(t_grid/us, curve, label=f'{T/uK:.2f} μK')
ax.set_xlabel('Release time [μs]')
ax.set_ylabel('Recapture probability')
ax.set_xlim(0, t_grid.max()/us)
ax.set_ylim(0, 1.05)
ax.legend()
plt.savefig('output/plot_simulated_curves.pdf', dpi=300, bbox_inches='tight')

# plot wavefunction expansion from defined initial state only
nr_times = np.shape(psi_x_evolved)[0]
times_to_plot = [0, nr_times//2, nr_times - 1]
fig1, ax1 = plt.subplots(figsize=(3.5, 2.8))
for t in times_to_plot:
    psi_x = psi_x_evolved[t, initial_state, :]
    prob = np.abs(psi_x)**2
    ax1.plot(x_grid/um, prob, label=fr'$t={t_grid[t]/us:.0f}$ μs')
ax1.set_xlabel('r [μm]')
ax1.set_ylabel(r'$|\psi(r)|^2$')
ax1.set_xlim(-2, 2)
ax1.legend()

plt.savefig('output/wavefunction_evolution.pdf', dpi=300, bbox_inches='tight')

plt.show()
