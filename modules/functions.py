import numpy as np
import pandas as pd
from scipy.constants import Boltzmann, pi, hbar
from numpy.fft import ifft, ifftshift, fftshift
import matplotlib.pyplot as plt

from modules.units import us, uK


def compute_r_squared(y_true, y_pred):
    """compute R^2 from the fit and the experimental data.

    Args:
        y_true (np.ndarray): the exp data
        y_pred (np.ndarray): the fit points

    Returns:
        r_squared (float): r^2 of fit 
    """
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res/ss_tot)
    return r_squared


def load_exp_data(data_path, data_name, survival_time):
    """load exp. data as csv file with header names 'Release time (us)', 'Surv. prob.', 'Error surv. prob.'
    rescale to 100% to account for imaging losses. Determined by taking first 20 us of so where
    the survival probability should be near 100%

    Args:
        data_path (str): 
        data_name (str): 
        survival_time (float): time where the surv. prob. is 100% in [s]

    Returns:
        exp_data_x: np array of x values 
        exp_data_y: np array of y values (survival probability)
        exp_data_yerr: np array of y values (error in survival probability)
    """
    exp_data = pd.read_csv(data_path + data_name)
    exp_data_x = exp_data['Release time (us)'].to_numpy()*us  # [s]
    exp_data_y = exp_data['Surv. prob.'].to_numpy()  # survival probability
    exp_data_yerr = exp_data['Error surv. prob.'].to_numpy()  # error in survival probability

    # rescale exp data to account for survival probability <100%
    indices = np.where((exp_data_x < survival_time*us))
    surv_prob = np.average(exp_data_y[indices])
    exp_data_y = exp_data_y/surv_prob
    exp_data_yerr = exp_data_yerr/surv_prob

    return exp_data_x, exp_data_y, exp_data_yerr


def prepare_grids(t_max, t_steps, x_max, nx):
    # --- Grids ---
    # Time grid for release times
    t_grid = np.linspace(0, t_max, t_steps)  # [s]

    # Spatial grid for wavefunction evaluation
    x_grid = np.linspace(-x_max, x_max, nx)
    dx = x_grid[1] - x_grid[0]

    # Momentum grid 
    k_grid = fftshift(np.fft.fftfreq(nx, d=dx)*2*pi)  # [rad/m]
    return t_grid, x_grid, dx, k_grid


def evolve_wavefunction(mass, t_grid, k_grid, k_basis_wf):
    """evolve wavefunction in momentum space using free evolution.

    Args:
        k_grid (np.ndarray): vector of momentum values
        k_basis_wf (np.ndarray): wavefunctions in momentum space (shape: (nr_states, nx))

    Returns:
        psi_x_evolved: np.ndarray, shape (nt, nr_states, nx)
    """

    # Phase factors for free evolution in momentum space
    phases = np.exp(-1j*(hbar*k_grid**2)/(2*mass)*t_grid[:, None])

    # Evolve all states at once: shape (nt, nr_states, nx)
    evolved_k = k_basis_wf[None, :, :]*phases[:, None, :]
    psi_x_evolved = ifft(ifftshift(evolved_k, axes=2), axis=2, norm='ortho')
    return psi_x_evolved


def compute_recapture_matrix(mass, t_grid, k_basis_wf, x_basis_wf, k_grid, dx):
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

    psi_x_evolved = evolve_wavefunction(mass, t_grid, k_grid, k_basis_wf)
    overlaps = dx*np.tensordot(psi_x_evolved, x_basis_wf.conj(), axes=([2], [1]))

    # Sum over final states to get recapture probabilities
    recap_prob = np.sum(np.abs(overlaps)**2, axis=2)  # shape (nt, nr_states)
    return recap_prob, psi_x_evolved


def compute_thermal_average(R_matrix, energies, temperatures):
    """
    Compute thermal averaged recapture curves.

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


def compute_recap_curves(mass, t_grid, basis_k, basis_x, k_grid, dx, energies, temperatures):
    """Compute both recapture probabilities and thermal averages.

    Args:
        basis_k: np.ndarray, wavefunctions in momentum space
        basis_x: np.ndarray, wavefunctions in position space
        k_grid: np.ndarray, momentum grid
        dx: float, spatial step size
        energies: np.ndarray, eigenenergies
        temperatures: np.ndarray, temperatures to simulate

    Returns:
        recap_matrix: shape (nt, nr_states)
        thermal_avg_curves: shape (len(temperatures), nt)"""
    
    recap_matrix, psi_x_evolved = compute_recapture_matrix(mass, t_grid, basis_k, basis_x, k_grid, dx)
    thermal_avg_curves = compute_thermal_average(recap_matrix, energies, temperatures)
    return thermal_avg_curves, psi_x_evolved
