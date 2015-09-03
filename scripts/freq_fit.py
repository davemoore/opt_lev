import numpy as np
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle
import opt_filt_util as ofu
import sys
import matplotlib
from scipy.interpolate import UnivariateSpline


#path to directory with chirps and charged bead
temp_path = "/data/20140717/Bead5/no_charge_rand_freq"

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/arider/analysis" + temp_path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

    



init_list = glob.glob(temp_path + "/*.h5")
files = sorted(init_list, key = bu.find_str)


init_dict = ofu.getdata_fft(files[0], temp_path)
dfft = init_dict['drive_fft']
rfft = init_dict['response_fft']

plt.loglog(np.abs(dfft))
plt.show()

for f in files[1::1]:
    try:

        curr_dict = ofu.getdata_fft(f, temp_path, response_column = 0)
         
        cdfft = curr_dict['drive_fft'] 
        crfft = curr_dict['response_fft']
        dfft += cdfft
        rfft += crfft
        #plt.loglog(np.abs(rfft))
        #plt.show()
    except:
        print "Holy Shit:",sys.exc_info()[0]


print 'lootp finished'



N_traces = len(files)-1

freqs = np.fft.rfftfreq(init_dict["N"], 1./init_dict['fsamp'])

#path to directory with chirps and charged bead
temp_path2 = "/data/20140717/Bead5/low_charge_chirps3"

## path to save plots and processed files (make it if it doesn't exist)
outpath2 = "/home/arider/analysis" + temp_path2[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)


init_list2 = glob.glob(temp_path2 + "/*.h5")
files2 = sorted(init_list2, key = bu.find_str)

init_dict2 = ofu.getdata_fft(files2[0], temp_path2)
dfft2 = init_dict2['drive_fft']
rfft2 = init_dict2['response_fft']

#plt.plot(np.abs(dfft))
in_drive = np.abs(dfft)>1e13

for f in files2[1::1]:
    try:

        curr_dict = ofu.getdata_fft(f, temp_path, response_column = 0)
        cdfft = curr_dict['drive_fft'] 
        crfft = curr_dict['response_fft']
        dfft2 += cdfft
        rfft2 += crfft
        #plt.loglog(np.abs(rfft))
        #plt.show()
    except:
        print "Holy Shit:",sys.exc_info()[0]


print 'lootp finished'

template = rfft2/(len(files2))
drive_freqs = freqs[in_drive]
inds = [np.argmin(np.abs(freqs - f)) for f in drive_freqs]

plt.loglog(freqs, np.abs(dfft))
plt.loglog(freqs[inds], np.abs(dfft[inds]), 'ro')
plt.show()

harmonics1 = 2.*freqs[in_drive]
inds1 = [np.argmin(np.abs(freqs - f)) for f in harmonics1]

harmonics2 = 3.*freqs[in_drive]
inds2 = [np.argmin(np.abs(freqs - f)) for f in harmonics2]

all_inds =[] #np.concatenate((inds, inds1, inds2))

plt.loglog(np.delete(freqs, all_inds), np.delete(np.abs(rfft), all_inds))
plt.loglog(freqs[inds], np.abs(rfft[inds]), 'or', label='Drive frequencies')
plt.loglog(freqs[inds1], np.abs(rfft)[inds1], 'ok', label='First harmonic')
plt.loglog(freqs[inds2], np.abs(rfft)[inds2], 'oc', label='Second harmonic')
plt.legend()
plt.show()

plt.loglog(np.delete(freqs, all_inds), np.delete(np.abs(rfft), all_inds))
plt.loglog(freqs[inds], np.abs(dfft[inds]), 'or', label='Drive frequencies')
plt.loglog(freqs[inds1], np.abs(dfft)[inds1], 'ok', label='First harmonic')
plt.loglog(freqs[inds2], np.abs(dfft)[inds2], 'oc', label='Second harmonic')
plt.legend()
plt.show()


plt.loglog(freqs[inds], np.abs(rfft[inds]/dfft[inds]), 'or', label='Drive frequencies')
plt.loglog(freqs[inds1], np.abs(rfft[inds1]/dfft[inds]), 'ok', label='First harmonic')
plt.loglog(freqs[inds2], np.abs(rfft[inds2]/dfft[inds]), 'oc', label='Second harmonic')
plt.legend()
plt.show()









ahat, sig_ahat, rcs = ofu.opt_filter2(ave_rfft[in_drive], template[in_drive], noise_psd[in_drive])

print ahat*20./500.









