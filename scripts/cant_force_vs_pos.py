## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab

#data_dir = "/data/20150908/Bead2/cant_mod"
data_dir = "/data/20150908/Bead2/chameleon"

NFFT = 2**19

conv_fac = 3.2e-14 ## N/V

def sort_fun( s ):
    cs = re.findall("_\d+.h5", s)
    return int(cs[0][1:-3])

flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)

tot_dat = []
for f in flist:

    print f 

    cdat, attribs, _ = bu.getdata( f )

    Fs = attribs['Fsamp']

    cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 
    
    plt.figure()
    plt.loglog( freqs, np.sqrt(cpsd)*conv_fac )
    plt.show()

    dtype = 0 if not "lowfb" in f else 1
    tot_dat.append( [cpos, np.sqrt(cpsd[freq_idx])*conv_fac*bw, dtype] )

tot_dat = np.array(tot_dat)

#tot_dat[:,0] = (80. - tot_dat[:,0]/10000.*80)

# old_dat = np.load("old_data.npy")

# print old_dat
# gpts = np.logical_and(old_dat[0] > 5, old_dat[0] % 2 == 0)
fig = plt.figure()
gpts = tot_dat[:,2] == 0
plt.semilogy(tot_dat[gpts,0], tot_dat[gpts,1], 'ks', label="High FB")
#gpts = tot_dat[:,2] == 1
#plt.semilogy(tot_dat[gpts,0], tot_dat[gpts,1], 'rs', label="Low FB")

# plt.errorbar(old_dat[0][gpts], old_dat[1][gpts], yerr=0.2*old_dat[1][gpts], fmt='ro', label="Old shielding")

#for i in range( len( tot_dat[:,0] ) ):
#    plt.gca().arrow( tot_dat[i,0], tot_dat[i,1], 0, -tot_dat[i,1]*0.5, fc='k', ec='k', head_width=1, head_length=0.1*tot_dat[i,1])

plt.legend(loc="upper right", numpoints=1)

#plt.xlim([0,50])
#plt.xlabel("Distance from cantilever [$\mu$m]")
plt.ylabel("Force [N]")

#fig.set_size_inches(6,4.5)

#plt.savefig("force_vs_dist.pdf")

plt.show()


