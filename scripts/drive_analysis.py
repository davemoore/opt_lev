
import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import bead_util as bu
from scipy.interpolate import UnivariateSpline as us


fname0 =  r"urmbar_xyzcool_close_3000mV_100Hz_1.h5"
fname1 =  r"urmbar_xyzcool_close_3000mV_100Hz_0.h5"
calf = r"../2mbar_zcool_50mV_41Hz.h5"


path = r"/data/20150331/Bead6/diploetap"

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

norm, bp, bcov = bu.get_calibration(os.path.join(path, calf), [1., 300.], make_plot = False)
#plt.show()

m = 1.6*10**-13
norm = norm*10**6
k = (2.*numpy.pi*bp[1])**2*m*10**-6



b, a = butter_bandpass(0.5, 250, fs)
x0 = ms(sp.filtfilt(b, a, dat0[:, 5], padlen = 150))
y0 = ms(dat0[:, 0]) #sp.filtfilt(b, a, dat0[:, 0], padlen = 150)

x1 = ms(numpy.around(sp.filtfilt(b, a, dat1[:, 5], padlen = 150), decimals = 4))
y1 = ms(dat1[:, 0])

plt.plot(x1)
plt.show()

s = spline(x0, y0)
xs = numpy.linspace(numpy.min(x0), numpy.max(x0), 5000)
ys = s(xs)
    

plt.plot(dist_scale(x0, dcal), norm*y0, '.', label = 'measured')
plt.plot(dist_scale(xs, dcal), norm*ys, 'r', linewidth = 2, label = 'interpolation')
plt.xlabel('Large bead displacement [um]')
plt.ylabel('Apparent transverse bead displacement from center of trap [um]')
plt.legend()
plt.show()

plt.plot(dist_scale(x1, dcal), norm*y1, '.', label = 'data 10 min later')
plt.plot(dist_scale(xs, dcal), norm*ys, 'r', linewidth = 2, label = 'original interpolating spline')
plt.xlabel('Large bead displacement [um]')
plt.ylabel('Apparent transverse bead displacement from center of trap [um]')
plt.show()

s2 = spline(x1, y1-s(x1), sm = 5)
xs2 = numpy.linspace(numpy.min(x1), numpy.max(x1), 5000)
ys2 = s2(xs2)

plt.plot(dist_scale(x1, dcal), k*norm*(y1-s(x1)), '.', label = 'residual force [N]')
plt.plot(dist_scale(xs2, dcal),k*norm*ys2, 'r', linewidth = 2, label = 'interpolated residual force [N]')
plt.xlabel('Large bead displacement [um]')
plt.ylabel('Apparent transverse force [N]')
plt.legend()
plt.show()


print numpy.std(k*norm*(y1-s(x1)))/numpy.sqrt(len(y1))

lc = xs2<0.
lm = numpy.mean(numpy.extract(lc,k*norm*(y1-s(x1))))

rc = xs2>0.
rm = numpy.mean(numpy.extract(rc,k*norm*(y1-s(x1))))

print rm-lm
