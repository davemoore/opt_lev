import h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import scipy.optimize as opt
import bead_util as bu
import numpy as np

refname = r"D:\Data\20150419\Bead2\urmbar_xyzcool_mod_fb_500mV_41Hz.h5"
dat, attribs, _ = bu.getdata(refname)

## first find the phase of the drive signal

b,a = sp.butter(3,0.1)

cdrive = np.abs(dat[:,3]-np.mean(dat[:,3]))
#cdrive = dat[:,3]
cdrive = sp.filtfilt( b, a, cdrive )
cdrive -= np.mean(cdrive)

cdat = dat[:,6]

def ffn(x, p0, p1, p2):
    return p0*np.sin( 2*np.pi*p1*xx + p2 ) 

## fit to a sin
xx = np.arange(len(cdrive))
spars = [np.std(cdrive)*2, 3.05e-4, 0]
bf, bc = opt.curve_fit( ffn, xx, cdrive, p0=spars )
#bf, bc = spars, ""

npts_per_cycle = 1./bf[1]
init_phase = (bf[2]/(2*np.pi) + 0.25 )*npts_per_cycle
#if( init_phase < 0 ):
#    init_phase += npts_per_cycle
print npts_per_cycle, init_phase

plt.figure()
plt.plot(xx[::4] ,dat[::4,3]-np.mean(dat[:,3]) )
plt.plot(xx[::4], cdrive[::4], 'k.' )
plt.plot(xx, ffn(xx, bf[0], bf[1], bf[2]), 'r' )
yy = plt.ylim()

tot_vec = np.zeros( (1,np.round( npts_per_cycle )) )
tot_cyc = int(len(cdrive)/np.round( npts_per_cycle ))
for n in range(0, tot_cyc ):

    sidx = np.round( n*npts_per_cycle - init_phase)
    eidx = sidx + np.round( npts_per_cycle )

    if( sidx < 0 ): continue
    if( eidx > len(cdat)): continue

    plt.plot( [sidx, sidx], yy, 'k' )

    tot_vec += cdat[sidx:eidx]

tot_vec = np.ndarray.flatten(tot_vec)
tot_vec -= np.mean(tot_vec)

b2,a2 = sp.butter(3,0.005)

filt_vec = sp.filtfilt(b2, a2, tot_vec)

plt.figure()
plt.plot(tot_vec)
plt.plot(filt_vec,'r')

filt_vec -= filt_vec[0]
filt_vec /= np.max(np.abs(filt_vec))

np.save("z_template.npy", filt_vec)


plt.show()

