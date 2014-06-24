## set of utility functions useful for analyzing bead data

import h5py, os, matplotlib, re
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as opt

bead_radius = 2.53e-6 ##m
bead_rho = 2.0e3 ## kg/m^3
kb = 1.3806488e-23 #J/K
bead_mass = 4./3*np.pi*bead_radius**3 * bead_rho

def getdata(fname):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        try:
            f = h5py.File(fname,'r')
            dset = f['beads/data/pos_data']
            dat = np.transpose(dset)
            max_volt = dset.attrs['max_volt']
            nbit = dset.attrs['nbit']
            dat = 1.0*dat*max_volt/nbit
            attribs = dset.attrs
        except KeyError:
            print "Warning, got no keys for: ", fname
            dat = []
            attribs = {}
            f = []
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []

    return dat, attribs, f

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
    

def inrange(x, xmin, xmax):
    return np.logical_and( x >= xmin, x<=xmax )

def bead_spec_rt_hz(f, A, f0, Damping):
    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0
    return np.sqrt(A*Damping/((omega_0**2 - omega**2)**2 + omega**2*Damping**2))


def get_calibration(refname, fit_freqs, make_plot=False, 
                    data_columns = [0,1], drive_column=-1, NFFT=2**14, exclude_peaks=False):
    ## given a reference file, fit the spectrum to a Lorentzian and return
    ## the calibration from V to physical units
    dat, attribs, cf = getdata(refname)
    if( len(attribs) > 0 ):
        fsamp = attribs["Fsamp"]
    xdat = dat[:,data_columns[0]]
    xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 

    ##first, fit for the absolute calibration
    damp_guess = 400
    f0_guess = 150
    Aemp = np.median( xpsd[fit_freqs[0]:fit_freqs[0]+10] )
    spars = [Aemp*(2*np.pi*f0_guess)**4/damp_guess, f0_guess, damp_guess]

    fit_bool = inrange( freqs, fit_freqs[0], fit_freqs[1] )

    ## if there's large peaks in the spectrum, it can cause the fit to fail
    ## this attempts to exclude them, assuming the first few points in the 
    ## range are clear of such peaks
    if(exclude_peaks):
        fit_bool = np.logical_and( fit_bool, xpsd < 1.5*Aemp )

    xdat_fit = freqs[fit_bool]
    ydat_fit = np.sqrt(xpsd[fit_bool])
    bp, bcov = opt.curve_fit( bead_spec_rt_hz, xdat_fit, ydat_fit, p0=spars)
    #bp = spars
    #bcov = 0.

    print bp

    print attribs["temps"][0]+273
    norm_rat = (2*kb*(attribs["temps"][0]+273)/(bead_mass)) * 1/bp[0]

    if(make_plot):
        fig = plt.figure()
        plt.loglog( freqs, np.sqrt(norm_rat * xpsd), '.' )
        plt.loglog( xdat_fit, np.sqrt(norm_rat * ydat_fit**2), 'k.' )
        xx = np.linspace( freqs[fit_bool][0], freqs[fit_bool][-1], 1e3)
        plt.loglog( xx, np.sqrt(norm_rat * bead_spec_rt_hz( xx, bp[0], bp[1], bp[2] )**2), 'r')
        plt.xlabel("Freq [Hz]")
        plt.ylabel("PSD [m Hz$^{-1/2}$]")
    
    return np.sqrt(norm_rat), bp, bcov


def find_str(str):
    """ Function to sort files.  Assumes that the filename ends
        in #mV_#Hz[_#].h5 and sorts by end index first, then
        by voltage """
    idx_offset = 1e10 ## large number to ensure sorting by index first

    fname, _ = os.path.splitext(str)

    endstr = re.findall("\d+mV_\d+Hz[_]?[\d+]*", fname)
    if( len(endstr) != 1 ):
        ## couldn't find the expected pattern, just return the 
        ## second to last number in the string
        return int(re.findall('\d+', fname)[-2])
        
    ## now check to see if there's an index number
    sparts = endstr[0].split("_")
    if( len(sparts) == 3 ):
        return idx_offset*int(sparts[2]) + int(sparts[0][:-2])
    else:
        return int(sparts[0][:-2])
    
