import numpy as np
import bead_util as bu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os


hot_path = '/data/20140801/Bead6'
hot_file = '2mbar_zcool_50mV_41Hz.h5'

cold_path = '/data/20140801/Bead6/plates_terminated'
cold_file = 'urmbar_xyzcool_50mV_RANDFREQ_4.h5'


def  ho_transfunc(freqs, A, f0, Q):
    #returns a harmonic oscillator transfeer function with resonant frequency fo and quality factor Q
    w0 = 2.*np.pi*f0
    sci = 1./(2.*Q)
    ws = 2.*np.pi*freqs
    return A/(-ws**2 + w0**2 + 2.j*sci*w0*ws)

def ho_psd(freqs, A, f0, Q):
    return np.abs(ho_transfunc(freqs, A, f0, Q))**2 

hot_dat, hot_attribs = bu.getdata(os.path.join(hot_path, hot_file))
cold_dat, cold_attirbs = bu.getdata(os.path.join(cold_path, cold_file))

hot_psds = np.abs(np.fft.rfft(hot_dat, axis = 0))**2
cold_psds = np.abs(np.fft.rfft(cold_dat, axis = 0))**2

freqs = np.fft.rfftfreq(len(hot_dat[:, 0]), 1./hot_attribs['Fsamp'])
b1 = freqs >10
b2 = freqs<500
bt = -b1-b2

p0h = [1.e9, 150., 1.]
p0c = [1e7, 150., 1.]

popt_xh, pcov_xh = curve_fit(ho_psd, freqs[bt], hot_psds[:, 0][bt], p0 = p0h)
popt_xc, pcov_xc = curve_fit(ho_psd, freqs[bt], cold_psds[:, 0][bt], p0 = p0c)

print popt_xh
print popt_xc
print popt_xc[0]/popt_xh[0]


plt.figure()
plt.subplot(2, 1, 1)
plt.loglog(freqs[bt], hot_psds[:, 0][bt], 'x')
plt.loglog(freqs[bt], cold_psds[:, 0][bt], 'o')
plt.loglog(freqs[bt], ho_psd(freqs[bt], popt_xh[0], popt_xh[1], popt_xh[2]), 'r')
plt.loglog(freqs[bt], ho_psd(freqs[bt], popt_xc[0], popt_xc[1], popt_xc[2]), 'c')
plt.subplot(2, 1, 2)
plt.loglog(freqs[bt], hot_psds[:, 1][bt])
plt.loglog(freqs[bt], cold_psds[:, 1][bt])

plt.show()
