import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os, glob, re
import scipy.signal as sp
import numpy as np
import bead_util as bu

def keyfunc(s):
        cs = re.findall("nmZ\d+nmZ", s)
        return int(cs[0][3:-3])

fname = "urmbar_xyzcool_stageX0nmY2500nmZ100nmZ500mVAC10Hz.h5"
#path = r"C:\Data\20150825\Bead1"
path = "/data/20150908/Bead2/cant_mod"

NFFT = 2**12		 

init_list = glob.glob(path + "/*xyzcool*500mVAC10Hz.h5")
files = sorted(init_list, key=keyfunc)

for f in files:
        try:
                data, attribs, handle = bu.getdata(f)
        except:
                print "Data read error"

        Fs = attribs['Fsamp']
        ypsd, freqs = mlab.psd(data[:,1], Fs=Fs, NFFT=NFFT)




'''
f = open('volt_to_newton_conv.txt')
conv = float(f.readline())

fig = plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(data0[0], np.sqrt(data0[1])*conv,label="Data")
if refname:
	plt.loglog(data1[0], np.sqrt(data1[1])*conv,label="Ref")
plt.ylabel("N/rt(Hz)")
#plt.ylabel("V$^2$/Hz")
plt.legend()
plt.subplot(3, 1, 2)
plt.loglog(data0[0], np.sqrt(data0[2])*conv)
if refname:
	plt.loglog(data1[0], np.sqrt(data1[2])*conv)
plt.subplot(3, 1, 3)
plt.loglog(data0[0], np.sqrt(data0[4]) * conv)
if refname:
	plt.loglog(data1[0], np.sqrt(data1[4]) * conv)
plt.ylabel("N/rt(Hz)")
#plt.ylabel("V$^2$/Hz")
plt.xlabel("Frequency[Hz]")
plt.show()
'''
