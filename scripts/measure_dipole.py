

## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle


temp_path = "/data/20150513/Bead1/2ftemp"
cal_path = "/data/20150513/Bead1/1ecal2f"
amp_path = "/data/20150513/Bead1/dipmeas_10um"
path  = "/data/20150513/Bead1"

dist_path = "/data/20150513/Bead1/dip_dist_speep"

## path to save plots and processed files (make it if it doesn't exist)
out_path = "/home/arider/analysis" + path[5:]
if( not os.path.isdir( out_path ) ):
    os.makedirs(out_path)

temp = bu.get_template(temp_path, 46, bw = 1.5)

def lin(x, m, b):
    return m*x +b

amps_cal, dcs_cal, drives_cal, dfreqs_cal = bu.amp_opt_filter_path(cal_path, temp)

p0 = [0.01, 0]

drives_cal = drives_cal*200*1e-3

bp, pcov = opt.curve_fit(lin, drives_cal, amps_cal, p0 = p0)

plt.plot(drives_cal, amps_cal, 'o')
plt.plot(drives_cal, lin(drives_cal, bp[0], bp[1]), 'r')
plt.show()

cal = 1.6e-17/bp[0]

amps_a, dcs_a, drives_a, dfreqs_a = bu.amp_opt_filter_path(amp_path, temp)
#amps_dc, dcs_dc, drives_dc, dfreqs_dc = bu.amp_opt_filter_path(dc_path, temp)

drives_a = 10e-4*drives_a

def parabola(x, a):
    return a*x**2

p0 = [0.03]

amps_a = np.abs(amps_a)*cal
bp, pcov = opt.curve_fit(parabola, drives_a, amps_a, p0 = p0)

plt.plot(drives_a, amps_a, 'x')
plt.plot(drives_a, parabola(drives_a, bp[0]), 'xr', linewidth = 2)
plt.show()

#plt.plot(dcs_dc, amps_dc, 'x')
#plt.plot(drives_a, parabola(drives_a, bp[0]), 'xr', linewidth = 2)
#plt.show()


dipole_coupling = (961978541345.0)

alpha = 2*bp[0]/dipole_coupling

print "measured alpha", alpha

rb = 2.5e-6
e0 = 9e-12
er = 5
chi = er - 1

alpha_th = 4./3.*np.pi*(rb)**3*e0*chi*(3./(er + 2.))#*1.5E15
print "We expect", alpha_th

amps_d, dcs_d, drives_d, dfreqs_d, dists_d = bu.amp_opt_filter_path(dist_path, temp, dist = True)

fs = amps_d*cal
distst = np.vstack((dists_d, fs))
np.save('distdata', distst)

plt.plot(dists_d, amps_d*cal, 'o')
plt.show()
