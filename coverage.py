#!/usr/bin/env python
#
# This script implements a Monte Carlo simulation to test the
# frequency of usable fields given modeled OIWFS probe geometries,
# while avoiding the IRIS image/IFU pickoffs, and given a surface
# density of potential guide stars.

import numpy as np
import oiwfs_sim
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Modifiable parameters
# --------------------------------------------------------------------------

# Establish the mean surface density of guide stars as a number of stars
# per 2'-diameter NFIRAOS circular FOV
num_per_FOV = 3.5
FOV_area_arcminsq = np.pi*(1.)**2.
density_perarcminsq = num_per_FOV/FOV_area_arcminsq

# number of Monte Carlo simulations
nmc = 1000

# Avoid the imager?
avoidImager = True

# --------------------------------------------------------------------------
# Start simulation
# --------------------------------------------------------------------------

# Create OIWFS State object
s = oiwfs_sim.State(None)
print "Are we avoiding the imager  :",avoidImager

# Outer loop over MC simulations
results = []
for i in range(nmc):
    if i % 100 == 0:
        print "%i / %i" % (i,nmc)
    # Ensure that no probe is assigned to a star from previous iteration
    for p in s.probes:
        p.star = None

    # Draw a Poisson-distributed samples of stars uniformly covering a 2'x2'
    # square centered over the origin of the OIWFS coordinate system
    l = (2.**2)*density_perarcminsq
    n = np.random.poisson(l)
    x_deg = np.random.uniform(low=-1.0/60.,high=1.0/60.,size=n)
    y_deg = np.random.uniform(low=-1.0/60.,high=1.0/60.,size=n)

    s.init_catalog(x_deg,y_deg)

    # try assigning probes to stars
    s.select_probes(catalog_subset=range(len(s.catalog_stars)), avoidImager=avoidImager)
    #for p in s.probes:
    #    if p.star is not None:
    #        # Shouldn't be necessary but may catch an error
    #        p.set_cart(p.star.x,p.star.y)
    
    nassigned = np.sum([p.park == False for p in s.probes])
    #print n, nassigned
    results.append([n,nassigned])

results = np.array(results)

print "Done:"
for i in range(4):
    num = np.sum(results[:,1]==i)
    percent = 100.*num/float(nmc)
    print "%i stars %i/%i = %.1f %%" % (i,num,nmc,percent)

