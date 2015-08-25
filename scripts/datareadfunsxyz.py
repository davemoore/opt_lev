import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np

refname = r"1mbar_xyzcool.h5"
fname0 = r""
path = r"C:\Data\20150825\Bead1"
d2plt = 1
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


		 

Fs = 5e3  ## this is ignored with HDF5 files
NFFT = 2**12

def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		
		#dat = 1.0*dat*max_volt/nbit

	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, 1]-numpy.mean(dat[:, 1]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, 2]-numpy.mean(dat[:, 2]), Fs = Fs, NFFT = NFFT)

	norm = numpy.median(dat[:, 2])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,2])**2
	return [freqs, xpsd, ypsd, dat, zpsd]

data0 = getdata(os.path.join(path, fname0))

def rotate(vec1, vec2, theta):
    vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
    vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
    return [vec1, vec2]


if refname:
	data1 = getdata(os.path.join(path, refname))
Fs = 10000
b, a = sp.butter(1, [2*5./Fs, 2*10./Fs], btype = 'bandpass')

if d2plt:	

        fig = plt.figure()
        plt.plot(data0[3][:, 0])
        plt.plot(data0[3][:, 1])
        plt.plot(data0[3][:, 3])
       # plt.plot(np.abs(data0[3][:, 3])-np.mean(np.abs(data0[3][:, 3])))
       



fig = plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(data0[0], data0[1],label="Data")
if refname:
	plt.loglog(data1[0], data1[1],label="Ref")
plt.ylabel("V$^2$/Hz")
plt.legend()
plt.subplot(3, 1, 2)
plt.loglog(data0[0], data0[2])
if refname:
	plt.loglog(data1[0], data1[2])
plt.subplot(3, 1, 3)
plt.loglog(data0[0], data0[4])
if refname:
	plt.loglog(data1[0], data1[4])
plt.ylabel("V$^2$/Hz")
plt.xlabel("Frequency[Hz]")
plt.show()
