import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

refname = "URmbar_xyzcool_nofilters3_elec3_250mV41Hz250mVdc_20.h5"
fname0 = "URmbar_xyzcool_nofilters3_elec3_250mV41Hz250mVdc_55.h5"
#path = r"C:\Data\20150825\Bead1"
path = "/data/20150909/Bead1/recharge_cal"
d2plt = 0
if fname0 == "":
	filelist = os.listdir(path)

	mtime = 0
	mrf = ""
	for fin in filelist:
		f = os.path.join(path, fin) 
		if os.path.getmtime(f)>mtime:
			mrf = f
			mtime = os.path.getmtime(f) 
 
	fname0 = mrf		

		 

NFFT = 2**12

def getpsd(data, attribs):
        Fs = attribs['Fsamp']
        xpsd, freqs = mlab.psd(data0[:,0], Fs=Fs, NFFT=NFFT)
        ypsd, freqs = mlab.psd(data0[:,1], Fs=Fs, NFFT=NFFT)
        zpsd, freqs = mlab.psd(data0[:,2], Fs=Fs, NFFT=NFFT)
        return [freqs, xpsd, ypsd, zpsd]

data0, attribs0, handle0 = bu.getdata(os.path.join(path, fname0))
freqs0, xpsd0, ypsd0, zpsd0 = getpsd(data0, attribs0)

if refname:
	data1, attribs1, handle1 = bu.getdata(os.path.join(path, refname))
        freqs1, xpsd1, ypsd1, zpsd1 = getpsd(data1, attribs1)

if d2plt:	

        fig = plt.figure()
        plt.plot(data0[:, 0])
        plt.plot(data0[:, 1])
        plt.plot(data0[:, 3])
       # plt.plot(np.abs(data0[3][:, 3])-np.mean(np.abs(data0[3][:, 3])))
       
f = open('volt_to_newton_conv.txt')
conv = float(f.readline())

print conv

fig = plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(freqs0, np.sqrt(xpsd0)*conv,label="Data")
if refname:
	plt.loglog(freqs1, np.sqrt(xpsd1)*conv,label="Ref")
plt.ylabel("N/rt(Hz)")
#plt.ylabel("V$^2$/Hz")
plt.legend()
plt.subplot(3, 1, 2)
plt.loglog(freqs0, np.sqrt(ypsd0)*conv)
if refname:
	plt.loglog(freqs1, np.sqrt(ypsd1)*conv)
plt.subplot(3, 1, 3)
plt.loglog(freqs0, np.sqrt(zpsd0) * conv)
if refname:
	plt.loglog(freqs1, np.sqrt(zpsd1) * conv)
plt.ylabel("N/rt(Hz)")
#plt.ylabel("V$^2$/Hz")
plt.xlabel("Frequency[Hz]")
plt.show()
