import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import numpy as np
import bead_util as bu

thresh = 10**7

path = "/data/20151023/trans_func/1_5mbar_zcool_fb2_synth1000mV2000Hz500mVdc.h5"

data, attribs, handle = bu.getdata(path)

chirp_fft = np.fft.rfft(data[:,-4])    # FFT of the artificial chirp
res_fft = np.fft.rfft(data[:,0])          # assuming chirp in X 
freqs = np.fft.rfftfreq(len(data[:,0]), 1. / attribs['Fsamp'])

power = chirp_fft * chirp_fft.conj()
inds = power > thresh

fb_tf = res_fft / chirp_fft





plt.figure(1)
plt.subplot(2,1,1)
plt.loglog(freqs[inds], np.abs(fb_tf[inds]))
plt.subplot(3,1,3)
#plt.figure(2)
plt.semilogx(freqs[inds], np.angle(fb_tf[inds]) * 180 / np.pi)
plt.xlim(0.1, 1000)
plt.figure(2)
plt.loglog(freqs[inds], power[inds], '.')
#plt.figure(4)
#plt.plot(data[:,-4])
plt.show()
