import numpy as np
from scipy.special import hermite
from scipy.constants import pi, hbar
import math
from math import log, exp, lgamma



class QuantumHarmonicOscillator:
    def __init__(self, omega, nr_states):
        self.omega = omega
        self.nr_states = nr_states
    
    def eigenenergies(self):
        """function to compute the eigenenergies of a harmonic oscillator

        Args:
            nr_states (int): number of states

        Returns:
            energies (ndarray): energies of the harmonic oscillator levels
        """
        
        energies = hbar*self.omega*(np.arange(self.nr_states) + 0.5)
        return energies

    def eigenstate(self, n, x, m):
        """function to compute the n-th eigenstate of a harmonic oscillator

        Args:
            x (float): position
            m (float): mass

        Returns:
            _type_: _description_
        """
        
        x_hermite = np.sqrt(m*self.omega/hbar)*x
        hermite_polynomial = hermite(x_hermite)
        log_norm = -0.5 * (n * log(2.0) + lgamma(n + 1)) + 0.25 * log(m * self.omega / (pi * hbar))
        norm = np.exp(log_norm)
        eigenfunction = norm*hermite_polynomial(x_hermite)*np.exp(-0.5*x_hermite**2)
        return eigenfunction


class GaussianPotential:
    def __init__(self, depth, waist):
        self.depth = depth
        self.waist = waist

    def calculate_nr_bound_states(self, m):
        """calculate nr of bound states in a Gaussian potential
        using the WKB approximation

        Args:
            m (float): mass

        Returns:
            nr_bound_states (int): number of bound states
        """
        
        nr_bound_states = self.waist/hbar*np.sqrt(2*m*self.depth/pi)
        return int(nr_bound_states)
    