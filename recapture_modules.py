import numpy as np
from scipy.special import hermite
from scipy.constants import pi, hbar
import math


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
        
        xi = np.sqrt(m*self.omega/hbar)*x
        hermite_polynomial = hermite(n)
        norm = 1.0/np.sqrt(2.0**n*math.factorial(n))*(m*self.omega/(pi*hbar))**0.25
        return norm*hermite_polynomial(xi)*np.exp(-0.5*xi**2)
