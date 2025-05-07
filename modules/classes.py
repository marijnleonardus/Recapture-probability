import numpy as np
from scipy.special import gammaln, eval_hermite
from scipy.constants import pi, hbar, Boltzmann, proton_mass
from numpy import log 
from numpy.fft import fft, fftshift
import pandas as pd

from modules.units import us


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
            n (int): harmonic oscillator level
            x (float): position
            m (float): mass

        Returns:
            eigenfunction: n-th eigenfunction of the harmonic oscillator 
        """
        
        # dimensionless position 
        xi = np.sqrt(m*self.omega/hbar)*x

        # compute normalization constant norm=1/(2^n*n!)(mω/πħ)^(1/4), but in log form to avoid large numbers

        # ln(norm) = -½ [n·ln2 + ln(n!)] + ¼·ln(m·ω/(π·ħ))
        # and using that gammaln(n+1) = ln(n!)
        # evaluate hermite and its sign and compute exp(psi) only at the end to avoid invalid numbers encoutering
        log_norm = (-0.5*(n*log(2) + gammaln(n + 1)) + 1/4*log(m*self.omega/(pi*hbar)))

        Hn_xi = eval_hermite(n, xi)
        sign_H = np.sign(Hn_xi)
        abs_H = np.abs(Hn_xi)

        log_psi = log_norm + np.log(abs_H) - 0.5 * xi**2

        # exponentiate safely; wherever abs_H==0, log_psi=-inf ⇒ exp→0
        psi = sign_H * np.exp(log_psi)
        return psi
        """ norm = np.exp(log_norm)

        #hermite_polynomial = hermite(n)
        norm = 1.0/np.sqrt(2.0**n*math.factorial(n))*(m*self.omega/(pi*hbar))**0.25
        eigenstate = norm*eval_hermite(n, xi)*np.exp(-0.5*xi**2)
        return eigenstate """


class GaussianPotential:
    def __init__(self, depth, omega):
        # initilize the Gaussian potential with depth and trap frequency in rad/s
        self.depth = depth  # J
        self.omega = omega  # rad/s

    def calculate_waist(self, m):
        """calculate the waist of a Gaussian potential from the depth and the trap freq

        Args:
            m (float): mass

        Returns:
            waist (float): waist of the Gaussian potential
        """
        
        waist  = np.sqrt(4*self.depth/(m*self.omega**2))
        return waist

    def calculate_nr_bound_states(self, m):
        """calculate nr of bound states in a Gaussian potential
        using the WKB approximation

        Args:
            m (float): mass

        Returns:
            nr_bound_states (int): number of bound states
        """
        
        waist = self.calculate_waist(m)
        nr_bound_states = waist/hbar*np.sqrt(2*m*self.depth/pi)
        return int(nr_bound_states)


class BoundStateBasis(GaussianPotential, QuantumHarmonicOscillator):
    def __init__(self, omega, mass, trap_depth, x_grid):
        self.omega = omega
        self.mass = mass
        self.trap_depth = trap_depth
        self.x_grid = x_grid

        # Initialize parent classes
        GaussianPotential.__init__(self, trap_depth*Boltzmann, omega)
        n_states = self.calculate_nr_bound_states(mass)
        print(f"Number of bound states: {n_states}")
        QuantumHarmonicOscillator.__init__(self, omega, n_states)

        self.n_states = n_states

    def prepare(self):
        x_basis_wf = np.array([
            self.eigenstate(n, self.x_grid, self.mass) for n in range(self.n_states)
        ])
        energies = self.eigenenergies()
        k_basis_wf = np.array([
            fftshift(fft(wf, norm='ortho')) for wf in x_basis_wf
        ])
        return x_basis_wf, k_basis_wf, energies    


def main():
    depth = 0.2*1e-3*Boltzmann  # J
    omega = 2*pi*54*1e3 
    mass = 85*proton_mass

    gaussian_potential = GaussianPotential(depth, omega)
    waist = gaussian_potential.calculate_waist(mass)
    print("Waist:", round(waist/1e-6, 2), "μm")

    nr_bound_states = gaussian_potential.calculate_nr_bound_states(mass)
    print("Number of bound states:", nr_bound_states)


if __name__ == "__main__":
    main()
