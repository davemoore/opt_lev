import numpy 
import bead_util as bu
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import os 

path  = '/data/20150428/Bead3'

cf = 'urmbar_xyzcool_mod_0_11V_2000mV_41Hz.h5'

dat, attribs, f = bu.getdata(os.path.join(cf, path))
fs = attribs['fsamp']
f.close


psd, freqs = ml.psd(dat[:, 0], NFFT = 2**17, Fs = fs)
plt.loglog(freqs, psd)
plt.show()
