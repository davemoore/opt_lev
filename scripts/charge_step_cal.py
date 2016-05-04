## load all files in a directory and using the correlation of the 
## of the response at the drive frequency, determine a force/volt
## calibration 

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

#path = r"D:\Data\20150202\Bead3\cantidrive\mon"
path = "/data/20150918/Bead1/chargelp_cal"
ts = 10.

fdrive = 41.
scale = 0.1  ## Guess the scaling of the response with a single e-
sig_to_noise = 0.2  ## Approximate value of the signal to noise in ampl

make_plot = True
reprocess_file = True

data_columns = [0, 1] ## column to calculate the correlation against
drive_column = 12 ## column containing drive signal


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
    



init_list = glob.glob(path + "/*xyzcool*_250mV*.h5")
print "SANITY"
files = sorted(init_list, key=keyfunc)
print files
for f in files[::1]:
        try:    
                cfile = f
                corr = getdata( cfile, 10. )
                corr_data.append(corr )
        except:
                print "uninformative error message"
    
corr_data = np.array(corr_data)
curr_corr = corr_data[:,0][0]

steps = []
k = 0
for point in corr_data:
        k += 1
        if k == len(files):
                break
        new_corr = corr_data[:,0][k]

        if np.abs( (new_corr - curr_corr) / curr_corr) > sig_to_noise:
                steps.append(k)
                if np.abs( new_corr - curr_corr ) > sig_to_noise:
                        break
        curr_corr = new_corr

step_vals = []
i = 0
for ind in steps:
        mean = np.mean(corr_data[:,0][i:ind])
        step_vals.append(mean)
        


print steps

'''
if make_plot:
        #nfiles = len(np.array(corr_data)[:,0])
        #t = np.linspace(0, nfiles-1, nfiles) * 10 
        plt.plot(np.array(corr_data)[:,0] / -0.15, linewidth=1.5, color='k')
        plt.grid()
        #plt.ylim(-5,5)
        #plt.xlabel("Time [s]")
        plt.ylabel("Bead response [# of e$^{-}$]")
        plt.show()
'''
 

    
