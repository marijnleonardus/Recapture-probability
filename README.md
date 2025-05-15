Script for modeling the probability of re-capturing a single atom after releasing in an optical tweezer.

If you want to fit your experimental data, use `fit_recapture_curve.py`.
If you just want to plot a bunch of theoretical curves, use `sim_recapture_curves.py`. 

Asisting functions and classes etc. are located in `modules/classes.py` and `modules/functions.py`

The working principle is the following: 
* first as the atom is trapped in an optical tweezer, it initilizes the wavefunction in a thermal distribution of harmonic oscillator states.
* Subsequently the trap is switched off and the wavefunction evolved under a free-space Hamiltonian
* The traps are switched on again: the overlap integral between the wavefunction and the bound harmonic oscillator states is computed

The number of bound states is approximately found using the WKB method for a Gaussian potential

Data is assumed to by a pandas dataframe in the form of a `.csv` file. If your data consists of individual np arrays instead, you can make it into a csv file using the script `process_raw_data.py`
