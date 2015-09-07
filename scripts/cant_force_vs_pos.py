## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab

#data_dir = "/data/20150903/Bead1/cant_drive"
data_dir = "/data/20150903/Bead1/cant_drive_elec_zeroed"

NFFT = 2**19

mod_freq = 3.05

#conv_fac = 1.6e-15/0.11 * (1./0.1) # N to V, assume 10 
conv_fac = 6.6e-13 * (1./0.1) # N to V 

def sort_fun( s ):
    return float(re.findall("Z\d+nm", s)[0][1:-2])

flist = sorted(glob.glob(os.path.join(data_dir, "urmbar_xyzcool_withmon_stageX0nmY2500nmZ*nmZ500mVAC10Hz.h5")), key = sort_fun)

tot_dat = []
for f in flist:

    cpos = float(re.findall("Z\d+nm", f)[0][1:-2])
    drive_freq = float(re.findall("\d+Hz.h5",f)[0][:-5])
    sig_freq = drive_freq + mod_freq

    print "Signal freq is: ", sig_freq, " Hz"

    cdat, attribs, _ = bu.getdata( f )

    Fs = attribs['Fsamp']

    cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 
    
    freq_idx = np.argmin( np.abs( freqs - sig_freq ) )

    bw = freqs[1]-freqs[0] ## bandwidth

    # plt.figure()
    # plt.semilogy( freqs, cpsd )
    # plt.semilogy( freqs[freq_idx], cpsd[freq_idx], 'rx' )
    # plt.xlim([sig_freq - 5, sig_freq + 5])
    # plt.show()

    tot_dat.append( [cpos, np.sqrt(cpsd[freq_idx])*conv_fac*bw] )

tot_dat = np.array(tot_dat)

tot_dat[:,0] = (80. - tot_dat[:,0]/10000.*80)

old_dat = np.load("old_data.npy")

print old_dat
gpts = np.logical_and(old_dat[0] > 5, old_dat[0] % 2 == 0)
fig = plt.figure()
plt.semilogy(tot_dat[:,0], tot_dat[:,1], 'ks', label="New shielding")
plt.errorbar(old_dat[0][gpts], old_dat[1][gpts], yerr=0.2*old_dat[1][gpts], fmt='ro', label="Old shielding")

for i in range( len( tot_dat[:,0] ) ):
    plt.gca().arrow( tot_dat[i,0], tot_dat[i,1], 0, -tot_dat[i,1]*0.5, fc='k', ec='k', head_width=1, head_length=0.1*tot_dat[i,1])

plt.legend(loc="upper right", numpoints=1)

plt.xlim([0,50])
plt.xlabel("Distance from cantilever [$\mu$m]")
plt.ylabel("Force [N]")

fig.set_size_inches(6,4.5)

plt.savefig("force_vs_dist.pdf")

plt.show()


