
import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt

import os
import scipy.signal as sp
import bead_util as bu
from scipy.interpolate import UnivariateSpline as us
import matplotlib.mlab 
import scipy.optimize as opt

fname0 =  r"urmbar_xyzcool_drive_5um_5000mV_9Hz.h5"
fname1 =  r"urmbar_xyzcool_drive_5um_5000mV_9Hz.h5"
calf = r"../2mbar_zcool_100mV_41Hz.h5"


path = r"/data/20150408/Bead2/drive"

dcal = 17.*5.


dat0, attribs0, f0 = bu.getdata(os.path.join(path, fname0))
fs = attribs0["Fsamp"]
f0.close()

dat1, attribs1, f1 = bu.getdata(os.path.join(path, fname1))
f1.close()

def dist_scale(x, dcal):
    return x*dcal/(numpy.max(x)-numpy.min(x))

def ms(dat):
    return dat-numpy.mean(dat)

def butter_bandpass(lowcut, highcut, fs, order=1):
     nyq = 0.5 * fs
     low = lowcut / nyq
     high = highcut / nyq
     b, a = sp.butter(order, high, btype='low')
     return b, a


def spline(x0, y0, sm = 1, dec = 4):
    x0 = numpy.around(x0, decimals = dec)
    xave = numpy.unique(x0)
    yave = numpy.zeros(len(xave))

    for i  in range(len(xave)):
        condition = x0 == xave[i]
        yave[i] = numpy.mean(numpy.extract(condition, y0))


    s = us(xave, yave, s = sm)
    return s

def bin_mean(xdat, ydat, bins):
    return (numpy.histogram(xdat, bins, weights = ydat)[0]/numpy.histogram(xdat, bins)[0])

def drive_func(x, a, f, p, a2, p2, o):
    return a*numpy.cos(f*x*2.*numpy.pi + p) + a2*numpy.cos(2.*f*x*2.*numpy.pi + p2) + o


norm, bp, bcov = bu.get_calibration(os.path.join(path, calf), [1., 300.], make_plot = True)
plt.show()

norm = norm*10**6
k = (2.*numpy.pi*bp[1])**2*bu.bead_mass*10**-6
dscale = 0.5

b, a = sp.butter(2, (0.0005, 0.001), btype = 'bandpass')

x0 = dat0[:, -1]*dscale
y0 = dat0[:, 0]*norm*k

x1 = dat1[:, -1]*dscale
y1 = dat1[:, 0]*norm*k


t = numpy.arange(len(x0))/fs

sp = numpy.array([5, 9, 0., 1., 0,  0.])

bp0, pcov0 = opt.curve_fit(drive_func, t, x0, sp)
bp1, pcov1 = opt.curve_fit(drive_func, t, x1, sp)


x0fit = drive_func(t, bp0[0], bp0[1], bp0[2], bp0[3], bp0[4], bp0[5])
x1fit = drive_func(t, bp1[0], bp1[1], bp1[2], bp1[3], bp1[4], bp1[5])

nbins = 100.
bins = numpy.linspace(numpy.min(x0), numpy.max(x0), nbins)



bmtemp = bin_mean(x0fit, y0, bins)
bmdata = bin_mean(x1fit, y1, bins)



plt.plot(t, x0, 'xr')
plt.plot(t, x1, 'xk')
plt.plot(t, x0fit, linewidth = 5)
plt.plot(t, x1fit, 'y',  linewidth = 5)
plt.show()

psd, freqs = matplotlib.mlab.psd(x0, NFFT = 2**19, Fs = 10000)
plt.loglog(freqs, psd)
plt.show()

#plt.plot(x0, y0, 'x')
plt.plot(bins[:-1], bmtemp, 'r', linewidth = 5, label = 's = 6um')
plt.plot(bins[:-1], bmdata, 'k', linewidth = 5, label = 's = 1um')
plt.legend()
plt.xlabel('cantilever displacement [um]')
plt.ylabel('Apparent force on bead [N]')
#plt.plot(x0, y0, 'x')
plt.show()

plt.plot(bins[:-1], bmdata-bmtemp, 'x', linewidth = 5)
plt.show()
