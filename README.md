Script for modeling the probability of re-capturing a single atom after releasing in an optical tweezer.

The main script is `model_recapture_probability.py` and assisting functions etc. are located in `modules.py`

The script works as follows: 
* first as the atom is trapped in an optical tweezer, it initilizes the wavefunction in a thermal distribution of harmonic oscillator states.
* Subsequently the trap is switched off and the wavefunction evolved under a free-space Hamiltonian
* The traps are switched on again: the overlap integral between the wavefunction and the bound harmonic oscillator states is computed

The number of bound states is approximately found using the WKB method for a Gaussian potential
