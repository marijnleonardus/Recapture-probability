# author: marijn venderbosch
# may 2025

# %% 

import numpy as np
from scipy.constants import pi, hbar, proton_mass, Boltzmann
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.integrate import simpson
from numpy.fft import fft, fftshift, ifft, ifftshift

# user defined libraries
from modules import QuantumHarmonicOscillator, GaussianPotential

us = 1e-6  # [s]
kHz = 1e3  # [Hz]
uK = 1e-6  # [K]

# variables
m = 85*proton_mass  # [kg]
trap_depth_K = 180*uK  # [K]
trap_freq = 54*kHz  # [Hz] trap frequency
temperatures = np.array([2*uK, 3*uK, 4*uK])  # [K]
t_release_max = 60*us  # [s] maximum release time
waist = 0.9e-6  # [m] waist of the optical tweezers

omega = 2*pi*trap_freq  # [rad/s] radial trap frequency

# %% 

# time grid
nr_timesteps = 30
t_grid = np.linspace(0, t_release_max, nr_timesteps)

# Spatial grid parameters
nr_x = 4096
x_max = 6e-6  # [m] spatial window half-width

# spatial grid
x = np.linspace(-x_max, x_max, nr_x)
dx = x[1] - x[0]

# k-space grid
k = fftshift(np.fft.fftfreq(nr_x, d=dx)*2*pi)

# estimate nr of bound states, divide U0 in Hz by the level spacing in Hz
trap_depth_Hz = trap_depth_K*Boltzmann/(hbar*2*pi)  # [Hz]
Tweezer = GaussianPotential(trap_depth_K*Boltzmann, waist)  # [K]
nr_bound_states = Tweezer.calculate_nr_bound_states(m)  
print(nr_bound_states)

# Build bound-state basis (first N_states)
QuantumHO = QuantumHarmonicOscillator(omega, nr_bound_states)
basis = np.array([QuantumHO.eigenstate(n, x, m) for n in range(nr_bound_states)])

# Energies of HO levels
energies = QuantumHO.eigenenergies()

# Precompute momentum-space forms for each basis state
# shape is (nr_bound_states, nr_x)
momentum_basis = np.array([
    fftshift(fft(basis[n], norm='ortho'))
    for n in range(nr_bound_states)
])


def comp_recap_single_n(n0):
    """compute the recapture probability as a function of time
    for a given initial level n0, n(t=0)

    Args:
        n0 (int): harmonic oscillator level n at t=0

    Returns:
        recapture (float): recapture probability
    """

    psi0_momentum = momentum_basis[n0]
    recapture = np.zeros_like(t_grid)

    for idx, t_final in enumerate(t_grid):
        phi_k_t = psi0_momentum*np.exp(-1j*(hbar*k**2)/(2*m)*t_final)
        psi_t = ifft(ifftshift(phi_k_t), norm='ortho')
        overlaps = np.array([
            simpson(y=np.conj(basis[n])*psi_t, x=x)
            for n in range(nr_bound_states)
        ])
        recapture[idx] = np.sum(np.abs(overlaps)**2)
    return recapture


def comp_therm_avg_recap(temp):
    # Probability of starting in starting state n is
    # p_n ∝ exp(-E_n / k_B T) and normalize
    weights = np.exp(-energies/(Boltzmann*temp))
    weights = weights/np.sum(weights)  

    # Compute all recapture curves in parallel
    curves = Parallel(n_jobs=-1)(delayed(comp_recap_single_n)(n0) for n0 in range(nr_bound_states))
    curves = np.array(curves)

    # Weighted average over levels
    avg_recapture = np.dot(weights, curves)
    return avg_recapture

# %% 

avg_recaptures_curves = []
for T in temperatures:
    # Compute recapture curves for each temperature
    avg_recapture_curve = comp_therm_avg_recap(T)
    avg_recaptures_curves.append(avg_recapture_curve)
avg_recaptures_curves = np.array(avg_recaptures_curves)

# %%

# Plotting the recapture probability
fig, ax = plt.subplots()
ax.grid()
labels = [f'{temp/uK} μK' for temp in temperatures]
ax.plot(t_grid/us, avg_recaptures_curves.T, label=labels)
#ax.plot(t_grid/us, avg_recapture_0K, label='0 K', linestyle='--')
ax.set_xlabel('Free evolution time [μs]')
ax.set_xlim(0, t_release_max/us)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Average recapture probability')
ax.set_title('Thermal-Averaged Recapture vs Free Evolution Time')
ax.legend()

plt.show()

# %%
