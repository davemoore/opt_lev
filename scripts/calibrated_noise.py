import numpy, h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

#path = r"C:\Data\20150825\Bead1"
path = "/data/20150903/Bead1/chargecal/urmbar_xyzcool.h5"

NFFT = 2**17

drive_freq = 41
drive_voltage = 80

trap_dim = 4e-3

e_charges = 1
e0 = 1.602e-19

	
data, attribs, _  = bu.getdata(path)
Fs = 5000 #attribs['Fsamp']

psd, freqs = mlab.psd(data[:,1], Fs=Fs, NFFT=NFFT)

index = np.argmin(np.abs(freqs - drive_freq))

drive_power = psd[index]

f = open('response_at_drive.txt', 'w')
f.write(np.sqrt(drive_power))
f.close()

     
force = (drive_voltage / trap_dim) * (e0 * e_charges)

conv = force / np.sqrt(drive_power)


