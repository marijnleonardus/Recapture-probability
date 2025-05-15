import numpy as np
import pandas as pd
from scipy.constants import  pi
from numpy.fft import fftshift

from modules.units import us


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
