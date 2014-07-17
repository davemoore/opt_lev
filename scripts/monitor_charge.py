## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = r"D:\Data\20140711\Bead6\chargelp_chirps_10s"
fdrive = 41.
make_plot = True

data_columns = [0, 1] ## column to calculate the correlation against
drive_column = -1 ## column containing drive signal

def getphase(fname):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        fsamp = attribs["Fsamp"]
        xdat = dat[:,data_columns[0]]

        xdat = np.append(xdat, np.zeros( int(fsamp/fdrive) ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
        maxv = np.argmax(corr2) 

        cf.close()

        print maxv
        return maxv


def getdata(fname, maxv):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat = dat[:,data_columns[0]]

        lentrace = len(xdat)
        ## zero pad one cycle
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr_full = np.correlate( xdat, dat[:,drive_column])/lentrace
        corr = corr_full[ maxv ]

        return corr

def get_most_recent_file(p):

    filelist = os.listdir(p)
    
    mtime = 0
    mrf = ""
    for fin in filelist:
        if( fin[-3:] != ".h5" ):
            continue
        f = os.path.join(path, fin) 
        if os.path.getmtime(f)>mtime:
            mrf = f
            mtime = os.path.getmtime(f)     

    return mrf


best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    plt.hold(False)

while( True ):
    ## get the most recent file in the directory and calculate the correlation

    cfile = get_most_recent_file( path )
    
    ## wait a sufficient amount of time to ensure the file is closed
    time.sleep(10)

    if( not best_phase ):
        best_phase = getphase( cfile )

    corr = getdata( cfile, best_phase )
    corr_data.append(corr)

    np.savetxt( os.path.join(path, "current_corr.txt"), [corr,] )

    if make_plot:
        plt.plot(corr_data)
        plt.draw()
        plt.pause(0.001)

    
