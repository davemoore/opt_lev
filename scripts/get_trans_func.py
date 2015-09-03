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
temp_path = "/data/20140717/Bead5/no_charge_chirps_other_comp_float_gnd"

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/arider/analysis" + temp_path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

    



init_list = glob.glob(temp_path + "/*.h5")
files = sorted(init_list, key = bu.find_str)


init_dict = ofu.getdata_fft(files[0], temp_path)
dfft = init_dict['drive_fft']
rfft = init_dict['response_fft']

#plt.plot(np.abs(dfft))


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



#plt.loglog(freqs[in_drive], np.abs(rfft[in_drive])/np.abs(dfft[in_drive]), '.')
#plt.show()


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
in_drive = np.abs(dfft2)>1e5

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


noise_spline = UnivariateSpline(freqs[-in_drive], np.abs(rfft[-in_drive]/len(files)))

plt.figure()
plt.loglog(freqs, np.abs(rfft)/len(files), 'r.')
plt.loglog(freqs, noise_spline(freqs), 'b')
plt.show()


fid = freqs[in_drive]


plt.errorbar(fid,  np.abs(rfft[in_drive]), fmt = 'o',yerr=noise_spline(fid))
plt.xscale("log")
plt.yscale("log")
plt.show()


plt.loglog(freqs[in_drive], np.abs(rfft[in_drive])*len(files2)*20./(len(files)*np.abs(dfft2[in_drive])*500.), '.')
plt.loglog(freqs[in_drive], np.abs(rfft2[in_drive])*500./(np.abs(dfft2[in_drive])*20.), 'r.')

plt.show()


