## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab

data_dir = "/data/20150903/Bead1/cant_elec_bias_drive_dcsweep"

NFFT = 2**19

mod_freq = 0 ##3.05

#conv_fac = 1.6e-15/0.11 * (1./0.1) # N to V, assume 10 
conv_fac = 6.6e-13 # N to V 

def sort_fun( s ):
    return float(re.findall("-?\d+mVdc", s)[0][:-4])

tot_0pot = []
for n in range(0,50):

    flist = sorted(glob.glob(os.path.join(data_dir, "urmbar_xyzcool_withmon_elec0_1000mV41Hz*mVdc_stageX0nmY2500nmZ9000nm_%d.h5"%n)), key = sort_fun)

    tot_dat = []
    for f in flist:

        cpos = sort_fun(f)
        drive_freq = float(re.findall("\d+Hz",f)[0][:-2])
        sig_freq = drive_freq + mod_freq

        #print "Signal freq is: ", sig_freq, " Hz, Vdc = ", cpos

        cdat, attribs, _ = bu.getdata( f )

        Fs = attribs['Fsamp']

        #cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 

        ## take correlation with drive and drive^2

        response = cdat[:, 1]
        drive = cdat[:, 7]
        drive2 = drive**2
        drive2 -= np.mean(drive2)

        corr_dr = bu.corr_func(drive, response, Fs, drive_freq)[0]
        corr_dr2 = bu.corr_func(drive2, response, Fs, drive_freq)[0]

        #freq_idx = np.argmin( np.abs( freqs - sig_freq ) )

        if( False ): ##xnp.abs( cpos ) < 50 ):
            cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 
            freq_idx = np.argmin( np.abs( freqs - sig_freq ) )

            plt.figure()
            plt.loglog( freqs, cpsd )
            plt.loglog( freqs[freq_idx], cpsd[freq_idx], 'rx' )
            #plt.xlim([sig_freq - 5, sig_freq + 5])
            plt.show()

        tot_dat.append( [cpos, corr_dr, corr_dr2 ] )

    tot_dat = np.array(tot_dat)

    #plt.figure()
    #plt.plot(tot_dat[:,0], tot_dat[:,1], 'ks', label = "Drive")
    #plt.plot(tot_dat[:,0], tot_dat[:,2], 'gs', label = "(Drive)$^2$")

    frange = [-1000, 1000]
    gpts = np.logical_and( tot_dat[:,0] > frange[0], tot_dat[:,0] < frange[1] )
    p = np.polyfit( tot_dat[gpts,0], tot_dat[gpts,1], 1 )

    #xx = np.linspace( frange[0], frange[1], 1e3 )

    #plt.plot(xx, np.polyval(p, xx), 'r', linewidth=1.5)

    print "Min force at: ", -p[1]/p[0]
    tot_0pot.append( -p[1]/p[0] )

    #plt.legend(loc="upper left", numpoints=1)

tot_0pot = np.array(tot_0pot)

plt.figure()
plt.plot(tot_0pot, 'k.-')
plt.show()


