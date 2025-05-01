import numpy as np
from scipy.constants import pi, hbar, proton_mass, Boltzmann
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.integrate import simpson
from numpy.fft import fft, fftshift, ifft, ifftshift

# user defined libraries
from recapture_modules import QuantumHarmonicOscillator

us = 1e-6  # [s]
kHz = 1e3  # [Hz]
uK = 1e-6  # [K]

# variables
m = 88*proton_mass  # [kg]
trap_depth_K = 50*uK  # [K]
trap_freq = 25*kHz  # [Hz] trap frequency
temperature = 0.73*uK  # [K]
t_release_max = 100*us  # [s] maximum release time

omega = 2*pi*trap_freq  # [rad/s] radial trap frequency

# time grid
t_grid = np.linspace(0, t_release_max, 100)

# Spatial grid parameters
nr_x = 1024
x_max = 6e-6  # [m] spatial window half-width

# spatial grid
x = np.linspace(-x_max, x_max, nr_x)
dx = x[1] - x[0]

# k-space grid
k = fftshift(np.fft.fftfreq(nr_x, d=dx)*2*pi)

# estimate nr of bound states, divide U0 in Hz by the level spacing in Hz
trap_depth_Hz = trap_depth_K*Boltzmann/(hbar*2*pi)  # [Hz]
nr_bound_states = int(np.floor(trap_depth_Hz/(trap_freq)))

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


def compute_recapture_curve(n0):
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


def compute_thermal_averaged_curve(temp):
    # Probability of starting in starting state n is
    # p_n ∝ exp(-E_n / k_B T) and normalize
    weights = np.exp(-energies/(Boltzmann*temp))
    weights = weights/np.sum(weights)  

    # Compute all recapture curves in parallel
    curves = Parallel(n_jobs=-1)(delayed(compute_recapture_curve)(n0) for n0 in range(nr_bound_states))
    curves = np.array(curves)

    # Weighted average over levels
    avg_recapture = np.dot(weights, curves)
    return avg_recapture


avg_recapture_T = compute_thermal_averaged_curve(temperature)
avg_recapture_0K = compute_thermal_averaged_curve(1e-9)

# Plotting the recapture probability
fig, ax = plt.subplots()
ax.grid()
ax.plot(t_grid/us, avg_recapture_T, label=f'{temperature/uK} μK')
ax.plot(t_grid/us, avg_recapture_0K, label='0 K', linestyle='--')
ax.set_xlabel('Free evolution time [μs]')
ax.set_xlim(0, t_release_max/us)
ax.set_ylim(0, 1)
ax.set_ylabel('Average recapture probability')
ax.set_title('Thermal-Averaged Recapture vs Free Evolution Time')
ax.legend()

plt.show()
