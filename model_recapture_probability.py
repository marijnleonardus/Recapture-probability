
# Author: Marijn Venderbosch 
# Date: May 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar, proton_mass, Boltzmann
from numpy.fft import fft, fftshift, ifft, ifftshift

# User-defined modules
from modules import QuantumHarmonicOscillator, GaussianPotential

# --- Physical Units ---
us = 1e-6  #  [s]
kHz = 1e3  #  [Hz]
uK = 1e-6  #  [K]

# --- System Parameters ---
mass = 85*proton_mass  # atom mass [kg]
trap_depth = 200*uK  # trap depth [K]
trap_frequency = 54*kHz  # trap frequency [Hz]

# --- Exp Data
x_data_string = 'data/0x.npy'
y_data_string = 'data/0av.npy'
yerr_data_string = 'data/0e.npy'

# --- Simulation parameters ---
temperatures = np.array([2.8,4])*uK
t_max = 60*us  # [s]
t_steps = 20  # number of time steps

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
    Returns:
      basis_wavefuncs: np.ndarray, shape (N_states, nx)
      momentum_basis: np.ndarray, shape (N_states, nx)
      energies: np.ndarray, shape (N_states,)
    """
    GaussianBeam = GaussianPotential(trap_depth*Boltzmann, trap_freq)
    n_states = GaussianBeam.calculate_nr_bound_states(mass)
    print(f"Number of bound states: {n_states}")

    QuantumHO = QuantumHarmonicOscillator(omega, n_states)
    basis_wavefuncs = np.array(
        [QuantumHO.eigenstate(n, x_grid, mass) for n in range(n_states)]
    )
    energies = QuantumHO.eigenenergies()

    # Momentum-space wavefunctions
    momentum_basis = np.array([
        fftshift(fft(wf, norm='ortho')) for wf in basis_wavefuncs
    ])

    return basis_wavefuncs, momentum_basis, energies


def compute_recapture_matrix(momentum_basis, basis_wavefuncs, k_grid, time_vals, dx, mass):
    """
    Vectorized evolution and overlap calculations:
      R[t_index, initial_state] = recapture probability
    """
    
    # Phase factors for free evolution in momentum space
    phases = np.exp(-1j*(hbar*k_grid**2)/(2*mass)*time_vals[:, None])

    # Evolve all states at once: shape (nt, nr_states, nx)
    evolved_k = momentum_basis[None, :, :]*phases[:, None, :]
    psi_x = ifft(ifftshift(evolved_k, axes=2), axis=2, norm='ortho')

    # Overlap integrals: <basis[m] | psi_x(t; n0)>
    overlaps = dx*np.tensordot(psi_x, basis_wavefuncs.conj(), axes=([2], [1]))

    # Sum over final states to get recapture probabilities
    recap_prob = np.sum(np.abs(overlaps)**2, axis=2)  # shape (nt, nr_states)
    return recap_prob


def compute_thermal_average(R_matrix, energies, temperatures):
    """
    Compute thermal average recapture curves.
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


def plot_sim(time_vals, avg_curves, temperatures):
    """
    Plot recapture probability vs. release time for each temperature.
    """
    plt.figure()
    for curve, T in zip(avg_curves, temperatures):
        plt.plot(time_vals/us, curve, label=f'{T/uK:.2f} μK')
    plt.xlabel('Release time [μs]')
    plt.ylabel('Recapture probability')
    plt.xlim(0, time_vals.max()/us)
    plt.ylim(0, 1.05)
    plt.grid(True)


def main():
    # load exp data
    x_data = np.load(x_data_string)*us
    y_data = np.load(y_data_string)
    yerr = np.load(yerr_data_string)

    # rescale exp data to account for survival probability <100%
    # by taking average of first data points (where curves are flat)
    indices = np.where((x_data < 5*us))
    surv_prob = np.average(y_data[indices])
    y_data = y_data/surv_prob
    yerr = yerr/surv_prob

    # simulate quantum model and plot result
    basis_x, basis_k, energies = prepare_basis(omega, mass, trap_depth, omega, x)
    recapture_prob_matrix = compute_recapture_matrix(basis_k, basis_x, k, time_vals, dx, mass)
    avg_curves = compute_thermal_average(recapture_prob_matrix, energies, temperatures)
    plot_sim(time_vals, avg_curves, temperatures)

    # plot experimental data
    plt.errorbar(x_data/us, y_data, yerr=yerr, markersize=3, fmt='o', capsize=5, label='Exp. data', color='navy')
    plt.legend()

    #plt.savefig('output/deeptraps.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
