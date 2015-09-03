## set of utility functions useful for analyzing bead data

import h5py, os, matplotlib, re, glob, sys
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sp
import copy
import generate_sweep as gs
from scipy.interpolate import UnivariateSpline

bead_radius = 2.53e-6 ##m
bead_rho = 2.0e3 ## kg/m^3
kb = 1.3806488e-23 #J/K
bead_mass = 4./3*np.pi*bead_radius**3 * bead_rho

## default columns for data files
data_columns = [0, 1] ## column to calculate the correlation against
drive_column = -1
laser_column = 3

def gain_fac( val ):
    ### Return the gain factor corresponding to a given voltage divider
    ### setting.  These numbers are from the calibration of the voltage
    ### divider on 2014/06/20 (in lab notebook)
    volt_div_vals = {0.:  1.,
                     1.:  1.,
                     20.0: 100./5.07,
                     40.0: 100./2.67,
                     80.0: 100./1.38,
                     200.0: 100./0.464}
    if val in volt_div_vals:
        return volt_div_vals[val]
    else:
        print "Warning, could not find volt_div value"
        return 1.


def get_dict(obj):
    #returns a dictionary with the keys and values from an hdf5 attributes object.
    out_dict = {}
    for k in obj.keys():
        out_dict.update({str(k):obj[k]})
    return out_dict             

def getdata(fname, other_comp = False):
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
            attribs = get_dict(dset.attrs)
            if other_comp:
                try:
                  path_lst = _.split('/')
                  path_lst[1] += '_slave'
                  fname_drive = '/'.join(path_lst) + '_drive'+ fext
                  
                  f_d = h5py.File(fname_drive,'r')
                  dset_drive = f_d['beads/data/pos_data']
                  dat_drive = np.transpose(dset_drive)
                  dat[:, -1] = dat_drive[:, 0]
                  #plt.plot(dat[:, -1])
                  #plt.show()
                  
                  f_d.close()
                except:
                  print 'well, shit' 
            ## correct the drive amplitude for the voltage divider. 
            ## this assumes the drive is the last column in the dset
            vd = attribs['volt_div'] if 'volt_div' in attribs else 1.0
            curr_gain = gain_fac(vd)
            dat[:,-1] *= curr_gain

            ## now double check that the rescaled drive amp seems reasonable
            ## and warn the user if not
            offset_frac = np.abs(np.sqrt(2)*np.std( dat[:,-1] )/(200.0 * attribs['drive_amplitude'] )-1.0)
            if( curr_gain != 1.0 and offset_frac > 0.1):
                print "Warning, voltage_div setting doesn't appear to match the expected gain for ", fname
            f.close()
        except (KeyError, IOError):
            print "Warning, got no keys for: ", fname
            dat = []
            attribs = {}
            f = []
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []
    
    return dat, attribs

def getdata_drive(fname):
    f = h5py.File(fname,'r')
    dset = f['beads/data/pos_data']
    max_volt = dset.attrs['max_volt']
    nbit = dset.attrs['nbit']
    dat = np.transpose(dset)*1.0*max_volt/nbit
    
    return dat, f

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
    dat, attribs = getdata(refname)
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
    ## this attempts to exclude them.  If a single boolean=True is passed,
    ## then any points 50% higher than the starting points are excluded (useful
    ## for th overdamped case). If a list defining frequency ranges is passed, e.g.:
    ## [[f1start, f1stop],[f2start, f2stop],...], then points within the given
    ## ranges are excluded
    if( isinstance(exclude_peaks, list) ):
        for cex in exclude_peaks:
            fit_bool = np.logical_and(fit_bool, np.logical_not( inrange(freqs, cex[0],cex[1])))
    elif(exclude_peaks):
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
    if(True):
        ## couldn't find the expected pattern, just return the 
        ## second to last number in the string
        return int(re.findall('\d+', fname)[-1])
        
    ## now check to see if there's an index number
    sparts = endstr[0].split("_")
    if( len(sparts) == 3 ):
        return idx_offset*int(sparts[2]) + int(sparts[0][:-2])
    else:
        return int(sparts[0][:-2])
    
def unwrap_phase(cycles):
    #Converts phase in cycles from ranging from 0 to 1 to ranging from -0.5 to 0.5 
    if cycles>0.5:
        cycles +=-1
    return cycles

def laser_reject(laser, low_freq, high_freq, thresh, N, Fs, plt_filt):
    #returns boolian vector of points where laser is quiet in band. Averages over N points.
    b, a = sp.butter(3, [2.*low_freq/Fs, 2.*high_freq/Fs], btype = 'bandpass')
    filt_laser_sq = np.convolve(np.ones(N)/N, sp.filtfilt(b, a, laser)**2, 'same')
    if plt_filt:
        plt.figure()
        plt.plot(filt_laser_sq)
        plt.plot(np.argwhere(filt_laser_sq>thresh),filt_laser_sq[filt_laser_sq>thresh],'r.')
        plt.show()
    return filt_laser_sq<=thresh


def good_corr(drive, response, fsamp, fdrive):
    corr = np.zeros(fsamp/fdrive)
    response = np.append(response, np.zeros( fsamp/fdrive-1 ))
    n_corr = len(drive)
    for i in range(len(corr)):
        #Correct for loss of points at end

        correct_fac = 1.0*n_corr/(n_corr-i)
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac
    return corr

def corr_func(drive, response, fsamp, fdrive, good_pts = [], filt = False, band_width = 1):
    #gives the correlation over a cycle of drive between drive and response.

    #First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response  = response - np.mean(response)

    #bandpass filter around drive frequency if desired.
    if filt:
        b, a = sp.butter(3, [2.*(fdrive-band_width/2.)/fsamp, 2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = sp.filtfilt(b, a, drive)
        response = sp.filtfilt(b, a, response)
    
    #Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

      
    #Throw out bad points if desired
    if len(good_pts):
        response[-good_pts] = 0.
        lentrace = np.sum(good_pts)    


    corr_full = good_corr(drive, response, fsamp, fdrive)/(lentrace*drive_amp**2)
    return corr_full

def corr_blocks(drive, response, fsamp, fdrive, good_pts = [], filt = False, band_width = 1, N_blocks = 20):
    #Computes correlation in blocks to determine error.

    #first determine average phase to use throughout.
    tot_phase =  np.argmax(corr_func(drive, response, fsamp, fdrive, good_pts, filt, band_width))
    
    #Now initialize arrays and loop over blocks
    corr_in_blocks = np.zeros(N_blocks)
    len_block = len(drive)/int(N_blocks)
    for i in range(N_blocks):
        corr_in_blocks[i] = corr_func(drive[i*len_block:(i+1)*len_block], response[i*len_block:(i+1)*len_block], fsamp, fdrive, good_pts, filt, band_width)[tot_phase]
    return [np.mean(corr_in_blocks), np.std(corr_in_blocks)/N_blocks]

def gauss_fun(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )

def orthonormalize(x, y, z):
    #performs graham-schmidt ortho-normalization on the vectors x, y, and z.
    xb = (x-np.mean(x))
    xb /= np.sqrt(np.sum(xb**2))
    yb = (y-np.mean(y))
    yb /= np.sqrt(np.sum(yb**2))
    zb = (z-np.mean(z))
    zb /= np.sqrt(np.sum(zb**2))
    y_ind = (yb - np.sum(yb*xb)*xb)
    y_ind /= np.sqrt(np.sum(y_ind**2)) 
    z_ind = (zb - np.sum(zb*xb)*xb - np.sum(zb*y_ind)*y_ind)
    z_ind /= np.sqrt(np.sum(z_ind**2))
    return xb, y_ind, z_ind
    
def getdata_ave(path, other_compu = False, condition = True):
    init_list = glob.glob(path + "/*RAND*.h5")
    files = sorted(init_list, key = find_str)
    #print files
    ave, attribs = getdata(files[3], other_comp = other_compu)
    for f in files[1:]:
        print f
        try:
            if condition:
                dat, attribs = getdata(f, other_comp = other_compu) 
                
                ave += dat
        except:
            print sys.exc_info()[0]
    
    return np.array(ave)/len(files) 
                
 
def amp_opt_filter(dat_fft, noise_psd, template_fft):
    freqs = np.fft.rfftfreq(500000, 1./5000)
    primes = gs.get_primes(200)[8:]
    in_drive = [np.argmin(np.abs(freqs-p)) for p in primes]
    dat_fft = dat_fft[in_drive]
    noise_psd = noise_psd[in_drive]
    template_fft = template_fft[in_drive]
    num = np.sum(np.conjugate(template_fft)*dat_fft/noise_psd)
    denom = np.sum(np.abs(template_fft)**2/noise_psd)
    return num/denom 

def amp_opt_filter_var(temp_fft, noise_psd, T, Fs):
    return 1./np.sum(2*T*np.abs(temp_fft)**2/noise_psd)
        


def amp_opt_filter_path(path, temp_fft, noise_psd, dat_column = 0, Allfiles = False):
    if Allfiles:
        init_list = glob.glob(path + "/*.h5")
    else:
        init_list = glob.glob(path + "/*RAND*.h5")
    files = sorted(init_list, key = find_str)
    out_dict = {'amplitudes':[]}
    for f in files:
        try:
            dat, attribs = getdata(f)
            dat_fft = np.fft.rfft(dat[:, dat_column])
            amp = amp_opt_filter(dat_fft, noise_psd, temp_fft)*10./(attribs['drive_amplitude']*0.2)
            print amp
            out_dict['amplitudes'].append(amp)
            
            for k in attribs.keys():
                if k in out_dict:
                    out_dict[k].append(attribs[k])
                else:
                    out_dict[k] = [attribs[k], ]
            
        except:
            print "SSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
        
    return out_dict

def get_noisepsd(fname):
    dat, attribs = getdata(fname)
    try:
        return np.abs(np.fft.rfft(dat, axis = 0))**2
    except:
        return []


def get_noisepsd_path(path):
    init_list = glob.glob(path + "/*.h5")
    files = sorted(init_list, key = find_str)
    psd  = get_noisepsd(files[0])
    N = 1.
    for f in files[1:]:
        psdt = get_noisepsd(f)
        if len(psdt) > 0:
            try:
                psd += psdt
                N += 1
            except:
                psd = psdt
                
    return psd/N

def get_template(cal_path, column = 0):
    #Given a path with steps in the charge, constructs a template normalized to one charge.
    #First construct a temporary template to determine the charge at each calibration data point.
    temptemp = np.fft.rfft(getdata_ave(cal_path)[:, column])
    amp_dict = amp_opt_filter_path(cal_path, temptemp, np.ones(len(temptemp)), dat_column = column)
    
    step_vals = np.abs(np.diff(amp_dict['amplitudes']))
    step_guess = np.mean(step_vals[step_vals > 1.*np.std(step_vals)])
    #Keep only non-zero points and singly charged points
    b1 = np.abs(amp_dict['amplitudes']) > 0.05*step_guess
    b2 = np.abs(amp_dict['amplitudes']) < 1.9*step_guess
        
    init_list = glob.glob(cal_path + "/*RAND*.h5")
    files = np.array(sorted(init_list, key = find_str))

    files = files[-b1-b2]
    print files
    ave, attribs = getdata(files[0])
    N = 1.
    for f in files[1:]:
        try:
            dat, attribs = getdata(f, other_comp = False) 
            ave += dat
            N += 1.
        except:
            print "SSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    print N
    return np.fft.rfft(ave[:, column])/N


def get_snr(psd, noise_psd):
    return np.sqrt(psd/noise_psd)



def get_forceangle(path, noise, freqs, drive_freq):
    init_list = glob.glob(path + "/*Hz*.h5")
    files = sorted(init_list, key = find_str)
    drive_ind = np.argmin(np.abs(freqs - drive_freq))
    thetas = []
    phis = []
    for f in files:
        try:
            dat, attribs = getdata(f) 
            psds = np.abs(np.fft.rfft(dat, axis = 0))**2
            fs = map()
            famp = np.sqrt(fs[0]**2 + fs[1]**2 + fs[2]**2)
            thetas.append(np.arccos(fs[2]/famp))
            phis.append(np.arctan(fs[1]/fs[0]))
        except:
            print "SSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    
    return thetas, phis



def get_forceangle_of(path, noise, temp):
    init_list = glob.glob(path + "/*RAND*.h5")
    files = sorted(init_list, key = find_str)
    thetas = []
    phis = []
    for f in files:
        try:
            dat, attribs = getdata(f) 
            ftx = np.fft.rfft(dat[:, 0])
            fty = np.fft.rfft(dat[:, 1])
            ftz = np.fft.rfft(dat[:, 2])
            fac = 10./(attribs['drive_amplitude']*0.2)
            fx = np.real(amp_opt_filter(ftx, noise[:, 0], temp))*fac
            fy = np.real(amp_opt_filter(fty, noise[:, 0], temp))*fac
            fz = np.real(amp_opt_filter(ftz, noise[:, 0], temp))*fac
            famp = np.sqrt(fx**2 + fy**2 + fz**2)
            
            thetas.append(np.arccos(fz/famp))
            phis.append(np.arctan(fy/fx))
        except:
            print "SSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    
    return thetas, phis
