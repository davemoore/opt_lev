import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu

path = '/data/20140805/Bead1/recharge'
cal_file = '/data/20140805/Bead1/2mbar_zcool_50mV_2500Hz.h5'

norm, bp, bcov  = bu.get_calibration(cal_file, [5, 300], make_plot = True)

t = np.arange(0., 100., 1./5000.)
freqs = np.fft.rfftfreq(500000, 1./5000.)

ave_dat = bu.getdata_ave(path)
ftabs = np.abs(np.fft.rfft(ave_dat, axis = 0))

plt.figure()
plt.subplot(2, 1, 1)
plt.loglog(freqs, ftabs[:, -1], label = 'Drive')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Drive [V/mm]')
plt.legend(loc = 'lower right')
plt.xlim([20, 250])
plt.subplot(2, 1, 2)
plt.loglog(freqs, ftabs[:, 0]*norm*1e6, label = 'X response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('X response [$\mu$ m]')
plt.legend(loc = 'lower right')
plt.xlim([20, 250])
plt.show()
