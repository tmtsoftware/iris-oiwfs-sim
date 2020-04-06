#!/usr/bin/env python
#
# Make plots from data created by oiwfs_sim

import numpy as np
import matplotlib.pyplot as plt

logdata = np.load('simulation.npz')
t = logdata['t']
probe_coords = logdata['probe_coords']
probe_targs = logdata['probe_targs']
oiwfs_coords = logdata['oiwfs_coords']

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)

ax.plot(t,probe_coords[:,2,0])
ax.plot(t,probe_targs[:,2,0])
ax.set_xlabel('Time (s)')

plt.show()
plt.close()
