import numpy as np
import seismic_noise as sn
import matplotlib.pyplot as plt
import bead_util as bu
import os

path = '/data/20140924/laserpower'

#psd, freqs = sn.getcsd_ave(path)
#plt.loglog(freqs, psd)
#plt.show()

dat, attribs = bu.getdata(os.path.join(path, 'sumon0_2500mV_no_synth_10.h5'))

plt.plot(dat[:, 0])
plt.show()
