#!/usr/bin/env python
#
# Make plots from data created by oiwfs_sim
#
# Things to figure out:
# - fraction of time w/ 3 NGS, 2 NGS, 1 NGS, 0 NGS.
# - mean time to re-acquire NGS when actively crab-walking
# - typical length of time for each configuration
# - run the sky at 0.1, 0.75, 0.5, 0.25 to see how things change

import numpy as np
import matplotlib.pyplot as plt

#logfile='simulation.npz'
#logfile='simulation_2mm_per_sec.npz'
#logfile='simulation_0.1arcmin_per_sec.npz'
#logfile='simulation_0.1arcsec_per_sec.npz'
#logfile='simulation_0.1arcsec_per_sec_trunc.npz'
#logfile='simulation_0.5arcsec_per_sec.npz'
logfile='simulation_1.0arcsec_per_sec.npz'

logdata = np.load(logfile)
t = logdata['t']
probe_coords = logdata['probe_coords']
probe_targs = logdata['probe_targs']
oiwfs_coords = logdata['oiwfs_coords']
if 'dt' in logdata:
    dt = logdata['dt']
else:
    dt = 0.05
n = len(t)

# Truncate arrays if full simulation didn't run
wherezero = np.argwhere(t==0)
if len(wherezero) > 1:
    last = wherezero[1][0] # upper range for slice that had data
    t = t[:last]
    probe_coords = probe_coords[:last,:,:]
    probe_targs = probe_targs[:last,:,:]
    oiwfs_coords = oiwfs_coords[:last,:]

    # Set to True to create truncated file
    if False:
        trunclogdata={
                'dt':dt,
                'probe_coords':probe_coords,
                'probe_targs':probe_targs,
                'oiwfs_coords':oiwfs_coords,
                't':t
            }

        trunclogfile = logfile.rsplit('.npz')[0]+'_trunc.npz'
        np.savez(trunclogfile,**trunclogdata)

# calculate and plot error signal for each probe
colors=['r','g','b','k']

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)

all_err = np.zeros((n,3))
all_state = np.zeros((n,3),dtype=int) # -1 if parked/parking, 0 ontarget, 1 moving

# Keep track of individual probe ontarget times and reconfig times
parkedtimes = []
ontargettimes = []
movingtimes = []

for i in range(3):
    err_x = probe_coords[:,i,0] - probe_targs[:,i,0]
    err_y = probe_coords[:,i,1] - probe_targs[:,i,1]
    err_abs = np.sqrt(err_x**2 + err_y**2)
    
    parked=np.isnan(err_abs)
    state = np.zeros(n)
    state[parked]=-1
    state[~parked]=err_abs[~parked]
    state[state>0]=1

    all_state[:,i] = state
    all_err[:,i] = err_abs

    startparked=None
    startontarget=None
    startmoving=None
    
    if state[0] == -1:
        startparked = 0
    elif state[0] == 1:
        startmoving = 0
    else:
        startontarget = 0

    for j in range(1,n):
        if state[j] != state[j-1]:
            if state[j-1] == -1:
                parkedtimes.append((j-startparked)*dt)
            elif state[j-1] == 0:
                ontargettimes.append((j-startontarget)*dt)
            else:
                movingtimes.append((j-startmoving)*dt)

            if state[j] == -1:
                startparked = j
            elif state[j] == 0:
                startontarget = j
            else:
                startmoving = j


    #ax.plot(t,err_abs,'--',color=colors[i])
    ax.plot(t,probe_targs[:,i,0],color=colors[i])
    ax.plot(t,probe_targs[:,i,1],color=colors[i])
    #ax.plot(t,oiwfs_coords[:,0],'k')

ax.set_xlabel('Time (s)')
plt.ylim((-250,220))
plt.show()
plt.close()

# Convert to numpy arrays
parkedtimes = np.array(parkedtimes)
ontargettimes = np.array(ontargettimes)
movingtimes = np.array(movingtimes)

# clip weird outliers
movingtimes=movingtimes[movingtimes<100]

# Calculate how many probes on-target (active in AO) over time

ontarget = all_state==0
numactive = np.sum(ontarget,axis=1)

t_total = n*dt

for i in range(4):
    t_active = len(numactive[numactive==i])*dt
    print "%i probes active for %.1f s / %.1f s (%.2f %%)" % \
        (i,t_active,t_total,100.*t_active/t_total)

# Traverse numactive and figure out how long each configuration is
# stable
config_times = []
startactive=numactive[0]
for i in range(1,n):
    if numactive[i] != numactive[i-1]:
        # configuration has changed. Record time.
        config_time = (i-startactive)*dt
        config_times.append(config_time)
        startactive=i
config_times = np.array(config_times)

config_mean = np.mean(config_times)
parked_mean = np.mean(parkedtimes)
moving_mean = np.mean(movingtimes)
ontarget_mean = np.mean(ontargettimes)


print "Means:\nstable config (all probes) %f s\nparked %f s\nmoving %f s\nontarget %f s" % \
    (config_mean, parked_mean, moving_mean, ontarget_mean)
    
rects = []
labels = [
    'all probes stable config ($\mu$=%.1fs)' % config_mean,
    'probe parked ($\mu$=%.1fs)' % parked_mean,
    'probe moving ($\mu$=%.1fs)' % moving_mean,
    'probe ontarget ($\mu$=%.1fs)' % ontarget_mean]
data = (config_times,parkedtimes,movingtimes,ontargettimes)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)

#binsize=10.
#bins = np.arange(0,np.max(ontargettimes),step=binsize)
bins = np.linspace(0,np.max(ontargettimes),num=20)
binsize=bins[1]-bins[0]
bincen = (bins[1:]+bins[:-1])/2.

for i in range(4):
    d = data[i]
    l = labels[i]

    h = np.histogram(d,bins,density=True)

    rect = ax.bar(
                bincen + i*binsize/4 - 2*binsize/4,
                h[0],
                width=binsize/5.,
                color=colors[i],
                linewidth=0)
    rects.append(rect[0])

ax.legend(rects, labels)

#ax.hist(config_times,100)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Density')
plt.title('Time distributions (sim length=%.1fhr, %i reconfigs)' % (t[-1]/3600,len(config_times)))
#plt.ylim((-10,250))
plt.show()
plt.close()