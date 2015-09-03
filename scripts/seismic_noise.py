import numpy as np
import scipy
import matplotlib.pyplot as plt
import bead_util as bu
import h5py, os, matplotlib, re, glob, sys

cal = 629 #[v/(m/s)]

path = '/data/20140912/Bead1/turbo_down/'

#def hotf(w, w0, k, Q):
 #   k = w0**2*m
  #  num = 1/k
   # denom = -w**2/w0**2 + 1.j*w*/(Q*w0) + 1
   # return num/denom


def getdata_gp(fname):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    try:
        f = h5py.File(fname,'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        max_volt = dset.attrs['max_volt']
        nbit = dset.attrs['nbit']
        dat = dat*max_volt/nbit
        attribs = bu.get_dict(dset.attrs)
        f.close()

    except (KeyError, IOError, TypeError):
        print "Warning, got no keys for: ", fname
        dat = []
        attribs = {}
        f = []
       
    return dat, attribs

def getvar_band(path, fmin, fmax):
    init_list = glob.glob(path + "/*.h5")
    #print init_list
    files = sorted(init_list, key = bu.find_str)
    files = files[:-1]
    varsh = np.zeros(len(files))
    varsv = np.zeros(len(files))
    #times = np.zeros(len(files))
    print files
    for i in range(len(files)):
        if i == 0:
            dat, attribs = getdata_gp(files[0])
            psdh, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = attribs['Fsamp'], NFFT = 2**10, detrend = matplotlib.mlab.detrend_mean)
            psdv, freqs = matplotlib.mlab.psd(dat[:, 1], Fs = attribs['Fsamp'], NFFT = 2**10, detrend = matplotlib.mlab.detrend_mean)
            minb = np.argmin(np.abs(freqs - fmin))
            maxb = np.argmin(np.abs(freqs - fmax))
            varsh[0] = np.sum(psdh[minb:maxb])            
            varsv[0] = np.sum(psdv[minb:maxb])            
        else:
            print files[i]
            dat, attribs = getdata_gp(files[i])
            psdh, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = attribs['Fsamp'], NFFT = 2**10, detrend = matplotlib.mlab.detrend_mean)
            psdv, freqs = matplotlib.mlab.psd(dat[:, 1], Fs = attribs['Fsamp'], NFFT = 2**10, detrend = matplotlib.mlab.detrend_mean)
            minb = np.argmin(np.abs(freqs - fmin))
            maxb = np.argmin(np.abs(freqs - fmax))
            varsh[i] = np.sum(psdh[minb:maxb])            
            varsv[i] = np.sum(psdv[minb:maxb])

    return varsh, varsv

def getpsd_ave(path):
    init_list = glob.glob(path + "/*.h5")
    #print init_list
    files = sorted(init_list, key = bu.find_str)
    files = files[0:150:10]
    #times = np.zeros(len(files))
    print files
    for i in range(len(files)):
        if i == 0:
            dat, attribs = getdata_gp(files[0])
            psdx, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            psdy, freqs = matplotlib.mlab.psd(dat[:, 1], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            psdz, freqs = matplotlib.mlab.psd(dat[:, 2], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            psdv, freqs = matplotlib.mlab.psd(dat[:, -1], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)

            psd = np.array([psdx, psdy, psdz, psdv])
            print files[i]
        else:
            print files[i]
            dat, attribs = getdata_gp(files[i])
            

            psdxt, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            psdyt, freqs = matplotlib.mlab.psd(dat[:, 1], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            psdzt, freqs = matplotlib.mlab.psd(dat[:, 2], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            psdvt, freqs = matplotlib.mlab.psd(dat[:, -1], Fs = attribs['Fsamp'], NFFT = 2**20, detrend = matplotlib.mlab.detrend_mean)
            
            psdt = np.array([psdxt, psdyt, psdzt, psdvt])
            psd += psdt
            
    return psd/(len(files)), freqs

def getcsd_ave(path):
    init_list = glob.glob(path + "/*.h5")
    #print init_list
    files = sorted(init_list, key = bu.find_str)
    files = files[0:112:1]
    #times = np.zeros(len(files))
    print files
    for i in range(len(files)):
        if i == 0:
            dat, attribs = getdata_gp(files[0])
            psd, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = attribs['Fsamp'], NFFT = 2**18, detrend = matplotlib.mlab.detrend_mean)
            print files[i]
        else:
            print files[i]
            dat, attribs = getdata_gp(files[i])
            psdt, freqs = matplotlib.mlab.psd(dat[:, 0], Fs = attribs['Fsamp'], NFFT = 2**18, detrend = matplotlib.mlab.detrend_mean)
            psd += psdt
            
    return psd/(len(files)), freqs

