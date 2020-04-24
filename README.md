# iris-oiwfs-sim

Basic Python simulation of IRIS OIWFS probe arm motion including collision avoidance. [This design document](https://docushare.tmt.org/docushare/dsweb/Get/Document-57345) describes the algorithm in detail.

oiwfs_sim.py:
  - Contains classes, variables and functions for running simulations. If the file is executed, see examples at the end where it says `if __name__ == '__main__':`
  - `class Probe` encodes the geometry of probes
  - `class State` includes a collection of 3 probes and deals with path planning, probe assignment etc.
  - `run_sim()` is a wrapper function for running simulations, creating figures
  - `oiwfs_sky()` is a helper function to experiment with observation planning. Given stars and probe assignments, the telescope pointing (all in RA, Dec), and an IRIS rotator position angle, print DS9 regions to the screen representing the probe stems and head.
  
plots.py:
  - If `oiwfs_sim.py` is executed it will write the results of a non-sidereal tracking simulation to a file called something like `simulation.npz`. plots.py reads in these results and produces plots representing the statitics of how long the probes spend in various configurations.
   
coverage.py:
  - This script executes a series of Monte Carlo simulations for randomly-selected fields to establish sky coverage statistics.
