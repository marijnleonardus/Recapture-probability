
# Author: Marijn Venderbosch 
# Date: May 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar, proton_mass, Boltzmann
from numpy.fft import fft, fftshift, ifft, ifftshift

# User-defined modules
from modules import QuantumHarmonicOscillator, GaussianPotential, Statistics

# --- Physical Units ---
us = 1e-6  #  [s]
um = 1e-6  #  [m]
kHz = 1e3  #  [Hz]
uK = 1e-6  #  [K]

# --- System Parameters ---
mass = 85*proton_mass  # atom mass [kg]
trap_depth = 200*uK  # trap depth [K]
trap_frequency = 54*kHz  # trap frequency [Hz]

# Raw data 
use_exp_data = True

if use_exp_data:
    # (release time in us, survival probability, error in survival probability)
    exp_data = pd.read_csv('data/sorted_data.csv')
    exp_data_x = exp_data['Release time (us)'].to_numpy()*us  # [s]
    exp_data_y = exp_data['Surv. prob.'].to_numpy()  # survival probability
    exp_data_yerr = exp_data['Error surv. prob.'].to_numpy()  # error in survival probability

    # rescale exp data to account for survival probability <100%
    indices = np.where((exp_data_x < 10*us))
    surv_prob = np.average(exp_data_y[indices])
    exp_data_y = exp_data_y/surv_prob
    exp_data_yerr = exp_data_yerr/surv_prob

    # --- Simulation parameters ---
    nr_temperatures = 10  # number of temperatures to simulate
    temperatures = np.linspace(2, 5, nr_temperatures)*uK  # [K]
    t_max = max(exp_data_x) #60*us  # [s]
    t_steps = len(exp_data_x)  #  20 #number of time steps
else:
    t_max = 60*us  # [s]
    t_steps =  40 #number of time steps

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
k = fftshift(np.fft.fftfreq(nx, d=dx)*2*pi)  # [rad/m]


def prepare_basis(omega, mass, trap_depth, trap_freq, x_grid):
    """
    Compute bound-state basis and energies.

    Args:
        omega: float, trap frequency [rad/s]
        mass: float, mass of the atom [kg]
        trap_depth: float, trap depth [K]
        trap_freq: float, trap frequency [Hz]
        x_grid: np.ndarray, spatial grid for wavefunction evaluation

    Returns:
        basis_wavefuncs: np.ndarray, shape (N_states, nx)
        momentum_basis: np.ndarray, shape (N_states, nx)
        energies: np.ndarray, shape (N_states,)
    """
    GaussianBeam = GaussianPotential(trap_depth*Boltzmann, trap_freq)
    n_states = GaussianBeam.calculate_nr_bound_states(mass)
    print(f"Number of bound states: {n_states}")

    QuantumHO = QuantumHarmonicOscillator(omega, n_states)
    basis_wavefuncs = np.array([QuantumHO.eigenstate(n, x_grid, mass) for n in range(n_states)])
    energies = QuantumHO.eigenenergies()

    # Momentum-space wavefunctions
    momentum_basis = np.array([fftshift(fft(wf, norm='ortho')) for wf in basis_wavefuncs])
    return basis_wavefuncs, momentum_basis, energies


def evolve_wavefunction(k_grid, momentum_basis):
    """evolve wavefunction in momentum space using free evolution.

    Args:
        k_grid (np.ndarray): vector of momentum values
        momentum_basis (np.ndarray): wavefunctions in momentum space (shape: (nr_states, nx))

    Returns:
        psi_x_evolved: np.ndarray, shape (nt, nr_states, nx)
    """

    # Phase factors for free evolution in momentum space
    phases = np.exp(-1j*(hbar*k_grid**2)/(2*mass)*time_vals[:, None])

    # Evolve all states at once: shape (nt, nr_states, nx)
    evolved_k = momentum_basis[None, :, :]*phases[:, None, :]
    psi_x_evolved = ifft(ifftshift(evolved_k, axes=2), axis=2, norm='ortho')
    return psi_x_evolved


def compute_recapture_matrix(momentum_basis, basis_wavefuncs, k_grid, dx):
    """
    Vectorized evolution and overlap calculations:

    Args:
        momentum_basis (np.ndarray): wavefunctions in momentum space (shape: (nr_states, nx))
        basis_wavefuncs (np.ndarray): bound-state wavefunctions in position space (shape: (nr_states, nx))
        k_grid (np.ndarray): vector of momentum values
        dx (float): grid spacing in position space

    Returns:
        recap_prob[t_index, initial_state] = recapture probability
    """

    psi_x_evolved = evolve_wavefunction(k_grid, momentum_basis)
    overlaps = dx*np.tensordot(psi_x_evolved, basis_wavefuncs.conj(), axes=([2], [1]))

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
        if T == 0.0:
        # if the temperature is zero, can't define Boltzmann distribution. 
        # But just set the weight of the lowest eigenfunctino to 1.
            weights = np.zeros_like(energies)
            weights[0] = 1
        else:
            weights = np.exp(-energies/(Boltzmann*T))
            weights = weights/weights.sum()
        avg_list.append(R_matrix.dot(weights))
    return np.array(avg_list)


def compute_best_fit(temperatures, avg_curves):

    r_squared_list = []
    for curve, T in zip(avg_curves, temperatures):
        rsquared = Statistics.compute_r_squared(exp_data_y, curve)
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


def plot_best_fit(recapture_prob_matrix, energies, fitted_temp):
    fig3, ax2 = plt.subplots()
    best_curve = compute_thermal_average(recapture_prob_matrix, energies, [fitted_temp])
    if use_exp_data:
        ax2.errorbar(exp_data_x/us, exp_data_y, yerr=exp_data_yerr, markersize=3, fmt='o', capsize=5, label='Exp. data', color='navy')
    ax2.plot(time_vals/us, best_curve[0,:], label=f'{fitted_temp/uK:.2f} μK')
    ax2.set_xlabel('Release time [μs]')
    ax2.set_ylabel('Recapture probability')
    ax2.set_xlim(0, time_vals.max()/us)
    ax2.set_ylim(0, 1.05)
    ax2.legend()


def main():
    # simulate quantum model and plot result
    basis_x, basis_k, energies = prepare_basis(omega, mass, trap_depth, omega, x)
    recapture_prob_matrix = compute_recapture_matrix(basis_k, basis_x, k, dx)
    avg_curves = compute_thermal_average(recapture_prob_matrix, energies, temperatures)
    fitted_temp = compute_best_fit(temperatures, avg_curves)*uK
    plot_best_fit(recapture_prob_matrix, energies, fitted_temp)
        
    #plt.savefig('output/deeptraps.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
