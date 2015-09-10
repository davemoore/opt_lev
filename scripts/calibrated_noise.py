import numpy, h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

#path = r"C:\Data\20150825\Bead1"
path = "/data/20150909/Bead1/recharge_cal/" \
       + "URmbar_xyzcool_nofilters3_elec3_250mV41Hz250mVdc_55.h5"

# Use NFFT < total number of points to avoid windowing
NFFT = 2**14

drive_freq = 41
drive_voltage = 50

trap_dim = 4e-3

e_charges = 1
e0 = 1.602e-19

	
data, attribs, _  = bu.getdata(path)
Fs = attribs['Fsamp']

ypsd, freqs = mlab.psd(data[:,1], Fs=Fs, NFFT=NFFT)
print freqs

binwidth = freqs[1]-freqs[0]

index = np.argmin(np.abs(freqs - drive_freq))

response_power = (ypsd[index] + ypsd[index-1] + ypsd[index+1]) * binwidth
     
force = (drive_voltage / trap_dim) * (e0 * e_charges)

conv = force / np.sqrt(response_power)

print conv, "(Newton / Volt)"

f = open('volt_to_newton_conv.txt', 'w')
f.write(str(conv))
f.close()

