## set of utility functions useful for analyzing bead data

import numpy as np
import h5py, os
import datetime as dt

def getdata(fname):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        f = h5py.File(fname,'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        max_volt = dset.attrs['max_volt']
        nbit = dset.attrs['nbit']
        dat = 1.0*dat*max_volt/nbit
        attribs = dset.attrs
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}

    return dat, attribs
                           

def labview_time_to_datetime(lt):
    ### Convert a labview timestamp (i.e. time since 1904) to a 
    ### more useful format (python datetime object)
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt
    
