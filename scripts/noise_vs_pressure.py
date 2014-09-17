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

path = "/data/20140617/Bead3/recharge_pramp"
cal_file = "/data/20140617/Bead3/2mbar_axcool_1mV_41Hz.h5"

NFFT = 2**15
data_columns = [0, 1] ## column to calculate the correlation against

## first get calibration to physical units
cal_fac, bp_cal, _ = bu.get_calibration(cal_file, [1,500], True)
plt.show()

## now step through the files and measure the noise at each pressure
init_list = glob.glob(path + "/*.h5")
files = sorted(init_list, key = bu.find_str)

 ## max pressure range for normalizing color scale
prange = [5e-7, 5e-2]
norm = matplotlib.colors.Normalize(vmin=np.log10(prange[0]), vmax=np.log10(prange[1]))
cmap = matplotlib.cm.jet
m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

def get_noise_at_pressure(dat):
    xpsd, f = matplotlib.mlab.psd(dat[:,data_columns[0]], Fs = attribs['Fsamp'], NFFT = NFFT) 
    xpsd = np.ndarray.flatten(xpsd)

    fit_freqs = [5., 35.]
    fit_idx = [np.argmin(np.abs(f-fit_freqs[0])), np.argmin(np.abs(f-fit_freqs[1]))]

    lin_fun = lambda x, A: x*0.0 + A
    bp, bcov = opt.curve_fit(lin_fun, f[fit_idx[0]:fit_idx[1]], np.sqrt(xpsd[fit_idx[0]:fit_idx[1]]), p0=xpsd[fit_idx[0]])

    print bp

    if(True):
        plt.figure()
        #plt.loglog(f, np.sqrt(xpsd), 'k')
        plt.loglog([f[fit_idx[0]], f[fit_idx[0]]], [bp[0], bp[0]], 'r')
        plt.show()

    return bp[0], np.sqrt( bcov[0,0] )

#plt.figure()
plist = []
for f in files[20:-1]:
    dat, attribs, cf = bu.getdata(os.path.join(path, f))
    p = attribs['pressures'][1]
    ## skip files with no recorded pressure
    if(p < 0): continue
    
    sfac, cbp, cbcov = bu.get_calibration(os.path.join(path, f), [5,500], False, exclude_peaks=True)
    #n, nerr = get_noise_at_pressure( dat )

    ## get fitted noise at 41 Hz
    n = bu.bead_spec_rt_hz(41., cbp[0], cbp[1], cbp[2])
    ##approximate (conservative) error
    neh = bu.bead_spec_rt_hz(41., cbp[0]+np.sqrt(cbcov[0,0]), 
                            cbp[1]-np.sqrt(cbcov[1,1]), cbp[2]+np.sqrt(cbcov[2,2]))

    nerr = np.abs( neh-n )

    plist.append( [n*cal_fac, nerr*cal_fac, p] )
    
    #ccol =  m.to_rgba(np.log10(p))
    #plt.loglog(f, xpsd, color=ccol)

    cf.close()

sfac, cbp, cbcov = bu.get_calibration( "/data/20140617/Bead3/0_2mbar_xyzcool_1mV_41Hz.h5", [5,500], False, exclude_peaks=True)
n = bu.bead_spec_rt_hz(41., cbp[0], cbp[1], cbp[2])
neh = bu.bead_spec_rt_hz(41., cbp[0]+np.sqrt(cbcov[0,0]), 
                         cbp[1]-np.sqrt(cbcov[1,1]), cbp[2]+np.sqrt(cbcov[2,2]))
nerr = np.abs( neh-n )
plist.append( [n*cal_fac, nerr*cal_fac, 0.2] )

sfac, cbp, cbcov = bu.get_calibration( "/data/20140617/Bead3/0_4mbar_xyzcool_1mV_41Hz.h5", [5,500], False, exclude_peaks=True)
n = bu.bead_spec_rt_hz(41., cbp[0], cbp[1], cbp[2])
neh = bu.bead_spec_rt_hz(41., cbp[0]+np.sqrt(cbcov[0,0]), 
                         cbp[1]-np.sqrt(cbcov[1,1]), cbp[2]+np.sqrt(cbcov[2,2]))
nerr = np.abs( neh-n )
plist.append( [n*cal_fac, nerr*cal_fac, 0.4] )

sfac, cbp, cbcov = bu.get_calibration(cal_file, [5,500], False, exclude_peaks=True)
n = bu.bead_spec_rt_hz(41., cbp[0], cbp[1], cbp[2])
neh = bu.bead_spec_rt_hz(41., cbp[0]+np.sqrt(cbcov[0,0]), 
                         cbp[1]-np.sqrt(cbcov[1,1]), cbp[2]+np.sqrt(cbcov[2,2]))
nerr = np.abs( neh-n )
plist.append( [n*cal_fac, nerr*cal_fac, 2] )

bead_floor_file = "/data/20140617/Bead3/6e-7mbar_xyzcool_nobead_0mV_41Hz.h5"
sfac, cbp, cbcov = bu.get_calibration(bead_floor_file, [5,500], False, exclude_peaks=True)
n = bu.bead_spec_rt_hz(41., cbp[0], cbp[1], cbp[2])
neh = bu.bead_spec_rt_hz(41., cbp[0]+np.sqrt(cbcov[0,0]), 
                         cbp[1]-np.sqrt(cbcov[1,1]), cbp[2]+np.sqrt(cbcov[2,2]))
nerr = np.abs( neh-n )
bead_floor =  [n*cal_fac, nerr*cal_fac, 40]
print bead_floor

plist = np.array(plist)
dp = np.abs(np.diff(plist[:,2])/plist[:-1, 2] )
dp = np.hstack( [0., dp] )


force_scale = (bu.bead_mass * (2*np.pi*bp_cal[1])**2)
emp_floor = 5e-10*force_scale

gpts = dp < 0.05
gpts[-3:] = True
fig=plt.figure()
plt.errorbar( plist[gpts,2], plist[gpts,0]*force_scale, yerr=plist[gpts,1]*force_scale,  fmt='k.', label="Measured")
ax = plt.gca()
xx = plt.xlim()
ax.set_xscale('log')
ax.set_yscale('log')
plt.loglog([1e-7, 10], [bead_floor[0]*force_scale, bead_floor[0]*force_scale], color=[0.5, 0.5, 0.5], label="Imaging noise") 
plt.loglog([1e-7, 10], [emp_floor, emp_floor], 'k--', label="Empirical floor") 
plt.xlabel("Pressure [mbar]")
plt.ylabel("Force sensitivity, $\sigma_F$, @ 41 Hz [N Hz$^{-1/2}$]")

## use libbrect expression for gas damping
p = np.logspace(-7, 1, 1e3) ## in mbar
p_torr = p/1.33322368
gamma1 = 2.*np.pi/(7. * bu.bead_radius/1e-6 * 10**-10/p_torr * (365.24*24*3600.))
sig_x = np.sqrt( 4*bu.kb*300.*gamma1*bu.bead_mass )
plt.plot( p, sig_x, 'r--', label="Pressure limited" )
plt.plot( p, sig_x+emp_floor, 'r' )

plt.legend(loc='upper left', numpoints=1)

fig.set_size_inches(8,6)
plt.savefig("noise_vs_pressure.pdf")

plt.show()

