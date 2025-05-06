import numpy as np
from scipy.special import gammaln, eval_hermite
from scipy.constants import pi, hbar, Boltzmann, proton_mass
from numpy import log 


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


class Statistics:
    @staticmethod
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
    

        print(waist)


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
