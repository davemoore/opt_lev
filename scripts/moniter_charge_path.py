## load all files in a directory and plot the correlation of the response
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

#path = r"D:\Data\20150202\Bead3\cantidrive\mon"
path = "/data/20150921/Bead1/chargelp_cal"
ts = 10.

fdrive = 41.
make_plot = True
reprocess_file = True

data_columns = [0, 1] ## column to calculate the correlation against
drive_column = 12 ## column containing drive signal

scaling = 0.155


def keyfunc(s):
	cs = re.findall("_\d+.h5", s)
	return int(cs[0][1:-3])


def getdata(fname, maxv):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat = dat[:,data_columns[1]]

        lentrace = len(xdat)
        ## zero pad one cycle
        corr_full = bu.corr_func( dat[:,drive_column], xdat, fsamp, fdrive)
        

        return corr_full[0], np.max(corr_full) 



best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    


if reprocess_file:

    init_list = glob.glob(path + "/*xyzcool*_250mV*.h5")
    print "SANITY"
    files = sorted(init_list, key=keyfunc)
    #print files
    for f in files[::1]:
        try:    
                cfile = f
                corr = getdata( cfile, 10. )
                corr_data.append(corr )
        except:
                print "uninformative error message"
    

    if make_plot:
	#nfiles = len(np.array(corr_data)[:,0])
	#t = np.linspace(0, nfiles-1, nfiles) * 10 
        plt.plot(np.array(corr_data)[:,0] / scaling, linewidth=1.5, color='k')
	plt.grid()
	#plt.ylim(-5,5)
        #plt.xlabel("Time [s]")
        plt.ylabel("Bead response [# of e$^{-}$]")
        plt.show()

 

    
