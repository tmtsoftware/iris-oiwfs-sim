#!/usr/bin/env python
#
# This script implements a Monte Carlo simulation to test the
# frequency of usable fields given modeled OIWFS probe geometries,
# while avoiding the IRIS image/IFU pickoffs, and given a surface
# density of potential guide stars.

import numpy as np
import oiwfs_sim
import matplotlib.pyplot as plt

# hack for debugging
#np.random.seed(0)

# --------------------------------------------------------------------------
# Modifiable parameters
# --------------------------------------------------------------------------

# Establish the mean surface density of guide stars as a number of stars
# per 2'-diameter NFIRAOS circular FOV
num_per_FOV = 3.5
FOV_area_arcminsq = np.pi*(1.)**2.
density_perarcminsq = num_per_FOV/FOV_area_arcminsq

print "**************************************************"
print "Catalog surface density: ",density_perarcminsq," /arcmin^2"
print "**************************************************"

# number of Monte Carlo simulations
nmc = 1000

# Avoid the imager?
avoidImager = True

# Plot simulation to screen? (have to close between each sim)
doPlot = False

# --------------------------------------------------------------------------
# Start simulation
# --------------------------------------------------------------------------

# Create OIWFS State object
s = oiwfs_sim.State(None,avoidImager=avoidImager)
print "Are we avoiding the imager  :",avoidImager

# Outer loop over MC simulations
print "Begin simulations..."
results = []
for i in range(nmc):
    if i % 100 == 0:
        print "%i / %i" % (i,nmc)

    # Ensure that no probe is assigned to a star from previous iteration,
    # and is initially assumed to be parked.
    for p in s.probes:
        p.star = None
        p.park = True

    # Draw a Poisson-distributed samples of stars uniformly covering a 2'x2'
    # square centered over the origin of the OIWFS coordinate system
    l = (2.**2)*density_perarcminsq
    n = np.random.poisson(l)
    x_deg = np.random.uniform(low=-1.0/60.,high=1.0/60.,size=n)
    y_deg = np.random.uniform(low=-1.0/60.,high=1.0/60.,size=n)

    # only keep stars that land within patrol FOV. Not strictly
    # required since select_probes() will also skip the ones
    # that land outside. However, this speeds things up because it
    # cuts down on the number of configurations that need to be
    # tested.
    r_mm = np.sqrt(x_deg**2+y_deg**2)*3600.*oiwfs_sim.platescale
    keep = (r_mm <= oiwfs_sim.r_patrol)
    x_deg=x_deg[keep]
    y_deg=y_deg[keep]
    n = len(x_deg)

    # Pass the catalog to the state object
    s.init_catalog(x_deg,y_deg)

    # Automatically assign probes to stars
    s.select_probes(catalog_subset=range(len(s.catalog_stars)))
    for p in s.probes:
        if (p.park is False) and (p.star is not None):

            # Shouldn't be necessary but may catch an error
            try:
                p.set_cart(p.star.x,p.star.y,avoidImager=avoidImager)
            except Exception:
                print "Shouldn't be here %i" % i
    
    # Count how many of the probes were successfully assigned to stars
    # and append to the array of results
    nassigned = np.sum([p.park == False for p in s.probes])
    #print n, nassigned
    results.append([n,nassigned])

    # Create a figure on the screen
    if doPlot:
        oiwfs_sim.run_sim(animate=None,catalog_xdeg=x_deg,catalog_ydeg=y_deg, \
            avoidImager=avoidImager, plotlim=[-150,150,-150,150], display=True)

# Display the results
results = np.array(results)
print "Done:"
for i in range(4):
    num = np.sum(results[:,1]==i)
    err = np.sqrt(num)
    percent = 100.*num/float(nmc)
    percent_err = 100.*err/float(nmc)
    print "%i stars %i/%i = %.1f +/- %.1f %%" % (i,num,nmc,percent,percent_err)

