
import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import bead_util as bu

refname =r"urmbar_xyzcool_50mV_100Hz.h5"
fname0 =  r"urmbar_xyzcool2_50mV_100Hz.h5"
calf = r"2mbar_zcool_500mV_21Hz.h5"
path = r"/data/20150501/Bead1"
d2plt = 0

norm, bp, bcov = bu.get_calibration(os.path.join(path, calf), [1., 400.], make_plot = True)

k = bp[1]**2*bu.bead_mass
cal = norm*k

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
NFFT = 2**14
def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		max_volt = dset.attrs['max_volt']
		nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		
		dat = 1.0*dat*max_volt/nbit

	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, 1], Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, 2], Fs = Fs, NFFT = NFFT)
	norm = numpy.median(dat[:, 2])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,2])**2
	return [freqs, numpy.sqrt(xpsd), numpy.sqrt(ypsd), dat, zpsd]

data0 = getdata(os.path.join(path, fname0))

def rotate(vec1, vec2, theta):
    vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
    vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
    return [vec1, vec2]


if refname:
	data1 = getdata(os.path.join(path, refname))
Fs = 5000
b, a = sp.butter(3, [2*10./Fs, 2*200./Fs], btype = 'bandpass')

if d2plt:	
	fig = plt.figure()
	#rotated = rotate(data0[3][:, 0],data0[3][:, 1], numpy.pi*(0))
	rotated = [data0[3][:,0], data0[3][:,1]]
        #plt.plot(rotated[0])
        #plt.plot(rotated[1])
        plt.plot(rotated[0], label = 'x')
        plt.plot(rotated[1], label = 'y')
        plt.plot(data0[3][:, 2], label = 'z')
        #plt.plot(data0[3][:, -1])
        #plt.plot(data0[3][:, 3], label = 'fucking laser')
        plt.legend()
       # plt.plot(data0[3][:, 3])
       # plt.plot(data0[3][:, 4])
        #plt.plot(data0[3][:, 5])
        #plt.plot(data0[3][:, -1])
        if refname:
            plt.plot(data1[3][:, 2],label='z ref')
            #plt.plot(data1[3][:, 1])

gf = data0[0]<2500.

fig = plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(data0[0][gf], cal*data0[1][gf],label="Data")
if refname:
	plt.loglog(data1[0][gf], cal*data1[1][gf],label="Ref")
plt.ylabel("N/sqrt(Hz)")
plt.legend()
plt.subplot(3, 1, 2)
plt.loglog(data0[0][gf], cal*data0[2][gf])
if refname:
	plt.loglog(data1[0][gf], cal*data1[2][gf])
plt.ylabel("N/sqrt(Hz)")
plt.subplot(3, 1, 3)
plt.loglog(data0[0], data0[4])
if refname:
	plt.loglog(data1[0], data1[4])
plt.ylabel("V$^2$/Hz")
plt.xlabel("Frequency[Hz]")
plt.show()
