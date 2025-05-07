
# Author: Marijn Venderbosch 
# Date: May 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar, proton_mass, Boltzmann
from numpy.fft import fft, fftshift, ifft, ifftshift

# User-defined modules
from modules import BoundStateBasis
from functions import compute_r_squared, load_exp_data
from units import us, kHz, uK, um

# System Parameters 
mass = 85*proton_mass  # atom mass [kg]
trap_depth = 200*uK  # trap depth [K]
trap_frequency = 54*kHz  # trap frequency [Hz]

# Simulation parameters
nr_temperatures = 10  # number of temperatures to simulate
temperatures = np.linspace(2, 5, nr_temperatures)*uK  # [K]
max_sim_time = 60*us  # [s] maximum simulation time
sim_time_steps = 40

# Raw data 
use_exp_data = False
data_name = 'sorted_data.csv'
unity_surv_time = 15  # [us] where exp data is not flat anymore, to account imaging losses

if use_exp_data:
    # (release time in us, survival probability, error in survival probability)
    exp_data_x, exp_data_y, exp_data_yerr = load_exp_data('data/', data_name, unity_surv_time)
    t_max = max(exp_data_x) #60*us  # [s]
    t_steps = len(exp_data_x)  #  20 #number of time steps
else:
    t_max = max_sim_time 
    t_steps =  sim_time_steps

# Derived quantities
omega = 2*pi*trap_frequency    # trap angular frequency [rad/s]

# --- Grids ---
# Time grid for release times
time_vals = np.linspace(0, t_max, t_steps)  # [s]

# Spatial grid for wavefunction evaluation
nx = 2048
x_max = 6e-6    # half-width [m]
x = np.linspace(-x_max, x_max, nx)
dx = x[1] - x[0]

# Momentum grid 
k_grid = fftshift(np.fft.fftfreq(nx, d=dx)*2*pi)  # [rad/m]


def evolve_wavefunction(k_grid, k_basis_wf):
    """evolve wavefunction in momentum space using free evolution.

    Args:
        k_grid (np.ndarray): vector of momentum values
        k_basis_wf (np.ndarray): wavefunctions in momentum space (shape: (nr_states, nx))

    Returns:
        psi_x_evolved: np.ndarray, shape (nt, nr_states, nx)
    """

    # Phase factors for free evolution in momentum space
    phases = np.exp(-1j*(hbar*k_grid**2)/(2*mass)*time_vals[:, None])

    # Evolve all states at once: shape (nt, nr_states, nx)
    evolved_k = k_basis_wf[None, :, :]*phases[:, None, :]
    psi_x_evolved = ifft(ifftshift(evolved_k, axes=2), axis=2, norm='ortho')
    return psi_x_evolved


def compute_recapture_matrix(k_basis_wf, x_basis_wf, k_grid, dx):
    """
    Vectorized evolution and overlap calculations:

    Args:
        k_basis_wf (np.ndarray): wavefunctions in momentum space (shape: (nr_states, nx))
        x_basis_wf (np.ndarray): bound-state wavefunctions in position space (shape: (nr_states, nx))
        k_grid (np.ndarray): vector of momentum values
        dx (float): grid spacing in position space

    Returns:
        recap_prob[t_index, initial_state] = recapture probability
    """

    psi_x_evolved = evolve_wavefunction(k_grid, k_basis_wf)
    overlaps = dx*np.tensordot(psi_x_evolved, x_basis_wf.conj(), axes=([2], [1]))

    # Sum over final states to get recapture probabilities
    recap_prob = np.sum(np.abs(overlaps)**2, axis=2)  # shape (nt, nr_states)
    return recap_prob


def compute_thermal_average(R_matrix, energies, temperatures):
    """
    Compute thermal average recapture curves.

    Args:
        R_matrix: np.ndarray, shape (nt, nr_states)
        energies: np.ndarray, shape (nr_states,)
        temperatures: np.ndarray, shape (n_temperatures,)

    Returns:
        avg_curves: np.ndarray, shape (len(temperatures), nt)
    """
    avg_list = []
    for T in temperatures:
        # this works even if T is a 1‑element array
        if np.allclose(T, 0.0):
            weights = np.zeros_like(energies)
            weights[0] = 1
        else:
            weights = np.exp(-energies / (Boltzmann * T))
            weights /= weights.sum()
        avg_list.append(R_matrix.dot(weights))
    return np.array(avg_list)


def compute_best_fit(temperatures, avg_curves):
    r_squared_list = []
    for curve, T in zip(avg_curves, temperatures):
        rsquared = compute_r_squared(exp_data_y, curve)
        r_squared_list.append(rsquared)
    r_squared_array = np.array(r_squared_list)

    fig2, ax1 = plt.subplots()
    ax1.scatter(temperatures/uK, r_squared_array, marker='o', color='navy')
    ax1.set_xlabel('Temperature [μK]')
    ax1.set_ylabel('R^2')

    # fit 3rd degree polynomial to the data and plot result
    coeffs = np.polyfit(temperatures/uK, r_squared_array, deg=3)
    plot_x_scale = np.linspace(min(temperatures/uK), max(temperatures/uK), 100)
    fitted_curve = np.polyval(coeffs, plot_x_scale)
    ax1.plot(plot_x_scale, fitted_curve, color='orange', label='Fit')

    # obtain best T 
    best_fit_temp = plot_x_scale[np.argmax(fitted_curve)]
    return best_fit_temp


def plot_fit(recapture_prob_matrix, energies, temps):
    """if you have exp data, show the best fit curve
    if you have no exp data, show cuves for all temperatures
     
    Args:
        recapture_prob_matrix (np.ndarray)
        energies (np.ndarray)
        temps (np.ndarray) """
    
    fig, ax = plt.subplots()
    for T in np.atleast_1d(temps):
        # compute the single-curve for this T
        curve = compute_thermal_average(recapture_prob_matrix, energies, [T])[0]
        ax.plot(time_vals/us, curve, label=f'{T/uK:.2f} μK')

    if use_exp_data:
        ax.errorbar(
            exp_data_x/us, exp_data_y, yerr=exp_data_yerr,
            fmt='o', capsize=5, label='Exp. data', color='navy'
        )

    ax.set_xlabel('Release time [μs]')
    ax.set_ylabel('Recapture probability')
    ax.set_xlim(0, time_vals.max()/us)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.show()


def main():
    basis_x, basis_k, wf_energies = BoundStateBasis(omega, mass, trap_depth, x).prepare()
    R = compute_recapture_matrix(basis_k, basis_x, k_grid, dx)

    if use_exp_data:
        avg = compute_thermal_average(R, wf_energies, temperatures)
        fitted_temp = compute_best_fit(temperatures, avg)*uK
        plot_fit(R, wf_energies, fitted_temp)
    else:
        plot_fit(R, wf_energies, temperatures)

    plt.show()
    plt.savefig('output/plot.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
