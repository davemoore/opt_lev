import numpy as np
import matplotlib
import bead_util as bu
import scipy
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob
import cPickle as pickle
import copy
import scipy.signal as sig

#Define functions and classes for use processing and fitting data.
def thermal_psd_spec(f, A, f0, g, n, s):
    #The position power spectrum of a microsphere normalized so that A = (volts/meter)^2*2kb*t/M
    w = 2.*np.pi*f #Convert to angular frequency.
    w0 = 2.*np.pi*f0
    num = g
    denom = ((w0**2 - w**2)**2 + w**2*g**2)
    return A*num/denom + n + s*w

def step_fun(x, q, x0):
    #decreasing step function at x0 with step size q.
    xs = np.array(x)
    return q*(xs<=x0)

def multi_step_fun(x, qs, x0s):
    #Sum of step functions for fitting charge step calibration to.
    rfun = 0.
    for i, x0 in enumerate(x0s):
        rfun += step_fun(x, qs[i], x0)
    return rfun

sf = lambda tup: tup[0] #Sort key for sort_pts.

def sort_pts(xvec, yvec):
    #sorts yvec and xvec to put in order of increasing xvec for plotting
    zl = zip(xvec, yvec)
    zl = sorted(zl, key = sf)
    xvec, yvec = zip(*zl)
    return np.array(xvec), np.array(yvec)

def emap(eind):
    # map from electrode number to data axis
    if eind == 1 or eind == 2:
        return 2
    elif eind == 3 or eind == 4:
        return 1
    elif eind == 5 or eind == 6:
        return 0


class Fit:
    #holds the optimal parameters and errors from a fit. Contains methods to plot the fit, the fit data, and the residuals.
    def __init__(self, popt, pcov, fun):
        self.popt = popt
        try:
            self.errs = np.diagonal(pcov)
        except ValueError:
            self.errs = "Fit failed"
        self.fun = fun

    def plt_fit(self, xdata, ydata, ax, scale = 'linear', xlabel = 'X', ylabel = 'Y', errors = []):
        xdata, ydata = sort_pts(xdata, ydata)
        #modifies an axis object to plot the fit.
        if len(errors):
            ax.errorbar(xdata, ydata, errors, fmt = 'o')
            ax.plot(xdata, self.fun(xdata, *self.popt), 'r', linewidth = 3)

        else:    
            ax.plot(xdata, ydata, 'o')
            ax.plot(xdata, self.fun(xdata, *self.popt), 'r', linewidth = 3)

        ax.set_yscale(scale)
        #ax.set_xscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([np.min(xdata), np.max(xdata)])
    
    def plt_residuals(self, xdata, ydata, ax, scale = 'linear', xlabel = 'X', ylabel = 'Residual', label = '', errors = []):
        #modifies an axis object to plot the residuals from a fit.
        xdata, ydata = sort_pts(xdata, ydata)
        if len(errors):
            ax.errorbar(xdata, self.fun(xdata, *self.popt) - ydata, errors, fmt = 'o')
        else:
            
            ax.plot(xdata, (self.fun(xdata, *self.popt) - ydata), 'o')
        
        #ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([np.min(xdata), np.max(xdata)])

    def css(self, xdata, ydata, yerrs, p):
        #returns the chi square score at a point in fit parameters.
        return np.sum((ydata))
        

    #def plt_chi_sq(self,xdata, ydata, errs, ax):
        #plots chi square contours. 

    


        

def thermal_fit(psd, freqs, fit_freqs = [10., 400.], kelvin = 300., fudge_fact = 1e-6, noise_floor = 0., noise_slope = 0.):
    #Function to fit the thermal spectra of a bead's motion
    #First need good intitial guesses for fit parameters.
    fit_bool = bu.inrange(freqs, fit_freqs[0], fit_freqs[1]) #Boolian vector of frequencies over which the fit is performed
    f0 = freqs[np.argmax(psd[fit_bool])] #guess resonant frequency from hightest part of spectrum
    df = freqs[1] - freqs[0] #Frequency increment.
    vpmsq = bu.bead_mass/(bu.kb*kelvin)*np.sum(psd[fit_bool])*df*len(psd)/np.sum(fit_bool) #Guess at volts per meter using equipartition
    g0 = 1./2.*f0 #Guess at damping assuming critical damping
    A0 = vpmsq*2.*bu.kb*kelvin/(bu.bead_mass*fudge_fact)
    p0 = [A0, f0, g0, noise_floor, noise_slope] #Initial parameter vectors 
    popt, pcov = curve_fit(thermal_psd_spec, freqs[fit_bool], psd[fit_bool], p0 = p0)
    if not np.shape(pcov):
        print 'Warning: Bad fit'
    f = Fit(popt, pcov, thermal_psd_spec)
    return f

def sbin(xvec, yvec, bin_size):
    #Bins yvec based on binning xvec into bin_size
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals /= fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    for i, b in enumerate(bins):
        idx = bins_vals == b
        y_binned[i] = np.mean(yvec[idx])
        y_errors[i] = scipy.stats.sem(yvec[idx])
    return bins, y_binned, y_errors

def sbin_pn(xvec, yvec, bin_size, vel_mult = 0.):
    #Bins yvec based on binning xvec into bin_size for velocities*vel_mult>0.
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals /= fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    if vel_mult:
        vb = np.gradient(xvec)*vel_mult>0.
        yvec2 = yvec[vb]
    else:
        vb = yvec == yvec
        yvec2 = yvec

    for i, b in enumerate(bins):
        idx = bins_vals[vb] == b
        y_binned[i] = np.mean(yvec2[idx])
        y_errors[i] = scipy.stats.sem(yvec2[idx])
    return bins, y_binned, y_errors


def get_h5files(dir):
    files = glob.glob(dir + '/*.h5') 
    files = sorted(files, key = bu.find_str)
    return files

def simple_loader(fname, sep):
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.close_dat(elecs=False)
    return fobj

def pos_loader(fname, sep):
    #Generate all of the position attibutes of interest for a single file. Returns a Data_file object.
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.ms()
    fobj.spatial_bin()
    fobj.close_dat()
    return fobj

def ft_loader(fname, sep):
    # Load files and computer FFTs. For testing out diagonalization
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.ms()
    fobj.get_fft()
    return fobj

def H_loader(fname, sep):
    #Generates transfer func data for a single file. Returns a Data_file object.
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.find_H()
    fobj.ms()
    fobj.close_dat(ft=False)
    return fobj

def sb_loader(fname, sep = [0,0,0], col = 1, find = 16):
    #loads the spacings between the drive frequency and the sidebands
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.ms()
    fobj.psd()
    b, a = sig.butter(4, 0.1, btype = 'high')
    psd = sig.filtfilt(b, a, np.ravel(fobj.psds[col]))
    f = fobj.psd_freqs
    df = fobj.electrode_settings[find]
    find = np.argmin((f - df)**2)
    lpsd = psd[:find]
    lf = f[:find]
    hpsd = psd[find:2*find]
    hf = -f[find:2*find] + 2.*df
    #plt.plot(lf, lpsd*hpsd)
    #plt.plot(hf, hpsd)
    #plt.plot(f[:find], rdpsd[:find]*(rdpsd[find:2*find][::-1]))
    #plt.plot(f, psd, 'o', markersize = 3)
    #plt.show()
    return fobj

#define a class with all of the attributes and methods necessary for processing a single data file to 
    

class Hmat:
    #this class holds transfer matricies between electrode drives and bead response.
    def __init__(self, finds, electrodes, Hmats):
        self.finds = finds #Indicies of frequences where there is an electrode being driven above threshold 
        self.electrodes = electrodes #the electrodes where there is statistically significant signal
        self.Hmats = Hmats #Transfer matrix at the frequencies 



class Data_file:
    #This is a class with all of the attributes and methods for a single data file.

    def __init__(self):
        self.fname = "Filename not assigned."
        #self.path = "Directory not assigned." #Assuming directory in filename
        self.pos_data = "bead position data not loaded"
        self.dc_pos = "DC positions not computed"
        self.binned_pos_data = "Binned data not computed"
        self.binned_data_errors = "bined data errors not computed"
        self.cant_data = "cantilever position data no loaded"
        self.binned_cant_data = "Binned cantilever data not computed"
        self.separation = "separation not entered"
        self.Fsamp = "Fsamp not loaded"
        self.Time = "Time not loaded"
        self.temps = "temps not loaded"
        self.pressures = "pressures not loaded"
        self.synth_setting = "Synth setting not loaded"
        self.dc_supply_setting = "DC supply settings not loaded"
        self.electrode_data = "electrode data not loaded yet"
        self.electrode_settings = "Electrode settings not loaded"
        self.electrode_dc_vals = "Electrode potenitals not loaded"
        self.stage_settings = "Stage setting not loaded yet"
        self.psds = "psds not computed"
        self.data_fft = "fft not computed"
        self.fft_freqs = "fft freqs not computed"
        self.psd_freqs = "psd freqs not computed"
        self.thermal_cal = "Thermal calibration not computed"
        self.H = "bead electrode transfer function not computed"
        self.noiseH = "noise electrode transfer function not computed"
        self.sb_spacing = "sideband spacing not computed."

    def load(self, fstr, sep, cant_cal = 8., stage_travel = 80., cut_samp = 2000, \
             elec_inds = [8, 9, 10, 12, 13, 14, 15]):
        #Methods to load the attributes from a single data file. sep is a vector of the distances of closes approach for each direction ie. [xsep, ysep, zsep] 
        dat, attribs, f = bu.getdata(fstr)
        
        self.fname = fstr
        
        dat = dat[cut_samp:, :]
        
        #Attributes coming from Labview Front pannel settings
        self.separation = sep #Manually entreed distance of closest approach
        self.Fsamp = attribs["Fsamp"] #Sampling frequency of the data
        self.Time = bu.labview_time_to_datetime(attribs["Time"]) #Time of end of file
        self.temps = attribs["temps"] #Vector of thermocouple temperatures 
        self.pressures = attribs["pressures"] #Vector of chamber pressure readings [pirani, cold cathode]
        self.synth_settings = attribs["synth_settings"] #Synthesizer fron pannel settings
        self.dc_supply_settings = attribs["dc_supply_settings"] #DC power supply front pannel testings.
        
        self.electrode_settings = attribs["electrode_settings"] #Electrode front pannel settings for all files in the directory.fist 8 are ac amps, second 8 are frequencies, 3rd 8 are dc vals 
        self.electrode_dc_vals = attribs["electrode_dc_vals"] #Front pannel settings applied to this particular file. Top boxes independent of the sweeps
        self.stage_settings = attribs['stage_settings'] #Front pannel settings for the stage for this particular file.
        
        #Data vectors and their transforms
        self.pos_data = np.transpose(dat[:, 0:3]) #x, y, z bead position
        self.dc_pos =  np.mean(self.pos_data, axis = -1)
        #self.pos_data = np.transpose(dat[:,[elec_inds[1],elec_inds[3],elec_inds[5]]])
        self.cant_data = np.transpose(np.resize(sep, np.shape(np.transpose(self.pos_data)))) + stage_travel - np.transpose(dat[:, 17:20])*cant_cal
        self.electrode_data = np.transpose(dat[:, elec_inds]) #Record of voltages on the electrodes

        f.close()

    def get_stage_settings(self, axis=2):
        if axis == 0:
            mask = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        elif axis == 1:
            mask = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        elif axis == 2:
            mask = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

        return self.stage_settings[mask]


    def ms(self):
        #mean subtracts the position data.
        ms = lambda vec: vec - np.mean(vec)
        self.pos_data  = map(ms, self.pos_data)


    def spatial_bin(self, bin_sizes = [1., 1., 4.], cant_axis = 2):
        #Method for spatially binning data based on stage z  position.
        
        self.binned_cant_data = [[[[], [], []], [[], [], []], [[], [], []]], \
                                 [[[], [], []], [[], [], []], [[], [], []]], \
                                 [[[], [], []], [[], [], []], [[], [], []]]] 
        self.binned_pos_data = [[[[], [], []], [[], [], []], [[], [], []]], \
                                [[[], [], []], [[], [], []], [[], [], []]], \
                                [[[], [], []], [[], [], []], [[], [], []]]]
        self.binned_data_errors = [[[[], [], []], [[], [], []], [[], [], []]], \
                                   [[[], [], []], [[], [], []], [[], [], []]], \
                                   [[[], [], []], [[], [], []], [[], [], []]]]

        for i, v in enumerate(self.pos_data):
            for j, pv in enumerate(self.cant_data):
                for si in np.arange(-1, 2, 1):
                    bins, y_binned, y_errors = \
                            sbin_pn(self.cant_data[j], v, bin_sizes[j], vel_mult = si)
                    self.binned_cant_data[si][i][j] = bins
                    self.binned_pos_data[si][i][j] = y_binned 
                    self.binned_data_errors[si][i][j] = y_errors 
        
        self.binned_cant_data = np.array(self.binned_cant_data)
        self.binned_pos_data = np.array(self.binned_pos_data)
        self.binned_data_errors = np.array(self.binned_data_errors)

    def psd(self, NFFT = 2**16):
        #uses matplotlib mlab psd to take a psd of the microsphere position data.
        psder = lambda v: matplotlib.mlab.psd(v, NFFT = NFFT, Fs = self.Fsamp)[0]
        self.psds = np.array(map(psder, self.pos_data))
        self.psd_freqs = np.fft.rfftfreq(NFFT, d = 1./self.Fsamp)

    def get_fft(self):
        #Uses numpy fft rfft to compute the fft of the position data
        self.data_fft = np.fft.rfft(self.pos_data)
        self.fft_freqs = np.fft.rfftfreq(np.shape(self.pos_data)[1])*self.Fsamp


    def thermal_calibration(self):
        #Use thermal calibration calibrate voltage scale into physical units
        #Check to see if psds is computed and compute if not.
        if type(self.psds) == str:
            self.psd()
            
        caler = lambda v: thermal_fit(v, self.psd_freqs) 
        self.thermal_cal = map(caler, self.psds)
    
    def plt_thermal_fit(self, coordinate = 0):
        #plots the thermal calibration and residuals
        if type(self.thermal_cal) == str:
            print "No thermal calibration"
        else:
            f, axarr = plt.subplots(2, sharex = True)
            fit_obj = self.thermal_cal[coordinate]
            fit_obj.plt_fit(self.psd_freqs, self.psds[coordinate], axarr[0]) 
            fit_obj.plt_residuals(self.psd_freqs, self.psds[coordinate], axarr[1])
            plt.show()

    
    def find_H(self, dpsd_thresh = 2e-2, mfreq = 1.):
        #Finds the phase lag between the electrode drive and the respose at a given frequency.
        #check to see if fft has been computed. Comput if not
        if type(self.data_fft) == str:
            self.get_fft()        
        
        dfft = np.fft.rfft(self.electrode_data) #fft of electrode drive in daxis. 
        
        N = np.shape(self.pos_data)[1]#number of samples
        dpsd = np.abs(dfft)**2*2./(N*self.Fsamp) #psd for all electrode drives
        
        inds = np.where(dpsd>dpsd_thresh)#Where the dpsd is over the threshold for being used.
        Hmatst = np.einsum('ij, kj->ikj', self.data_fft, 1./dfft) #transfer matrix between electrodes and bead motion for all frequencies
        finds = inds[1] #frequency index with significant drive
        cinds = inds[0] #colun index with significant drive

        b = finds>np.argmin(np.abs(self.fft_freqs - mfreq))

        data_psd = np.abs(self.data_fft)**2*2./(N*self.Fsamp)

        dat_ind = emap(cinds[b][0])
        #plt.loglog(self.fft_freqs, dpsd[cinds[b][0]])
        #plt.loglog(self.fft_freqs, data_psd[dat_ind])
        #plt.show()

        #print cinds[b][0]

        # roll the response fft to compute a noise H
        shift = int(0.5 * (finds[b][1]-finds[b][0])) 
        randadd = np.random.choice(np.arange(-int(0.1*shift), int(0.1*shift)+1, 1))
        shift = shift+randadd

        rolled_data_fft = np.roll(self.data_fft, shift, axis=-1)

        #print finds[b]
        #print shift_for_noise
        #raw_input()

        Hmatst_noise = np.einsum('ij, kj->ikj', rolled_data_fft, 1./dfft)

        self.H = Hmat(finds[b], cinds[b], Hmatst[:, :, finds[b]])
        self.noiseH = Hmat(finds[b], cinds[b], Hmatst_noise[:, :, finds[b]])


    def plt_psd(self, col = 1):
        #plots psd
        #b, a = sig.butter(4, 0.01, btype = 'highpass')
        plt.loglog(self.psd_freqs, self.psds[col], label = str(self.electrode_settings[24]))

    def close_dat(self, p = True, psd = True, ft = True, elecs = True):
        #Method to reinitialize the values of some lage attributes to avoid running out of memory.
        if ft:
            self.data_fft = 'fft cleared'
            self.fft_freqs = 'fft freqs cleared'

        if psd:
            self.psds = 'psds cleared'
            self.psd_freqs = 'psd freqs cleared'

        if elecs:
            self.electrode_data = 'electrode data cleared'
        
        if p:
            self.cant_data = 'cantilever position data cleared'
            self.pos_data = 'bead position data cleared'



#Define a class to hold information about a whole directory of files.
class Data_dir:
    #Holds all of the information from a directory of data files.

    def __init__(self, paths, sep, label):
        all_files = []
        for path in paths:
            all_files =  (all_files + get_h5files(path))[:]
        self.label = label
        self.files = sorted(all_files, key = bu.find_str)
        self.sep = sep
        self.fobjs = "Files not loaded"
        self.Hs = "Transfer functions not loaded"
        self.noiseHs = "Noise Transfer functions not loaded"
        self.Havg = "Havg not computed"
        self.thermal_calibration = "No thermal calibration"
        self.charge_step_calibration = "No charge step calibration"
        self.ave_force_vs_pos = "Average force vs position not computed"
        self.ave_pos_data = "Average response not computed"
        self.ave_dc_pos = "Mean positions not computed"
        self.ave_pressure = 'pressures not loaded'
        self.paths = paths
        self.out_path = path.replace("/data/","/home/charles/analysis/")
        if len(self.files) == 0:
            print "Warning: empty directory"


    def load_dir(self, loadfun, maxfiles=10000, save_dc=False):
        #Extracts information from the files using the function loadfun which return a Data_file object given a separation and a filename.
        l = lambda fname: loadfun(fname, self.sep)
        self.fobjs = map(l, self.files[:maxfiles])
        per = lambda fobj: fobj.pressures
        self.ave_pressure = np.mean(map(per, self.fobjs), axis = 0)

        
        self.ave_dc_pos = np.zeros(3)
        count = 0
        for obj in self.fobjs:
            if type(obj.dc_pos) != str:
                self.ave_dc_pos += obj.dc_pos
                count += 1
        if count:
            self.ave_dc_pos = self.ave_dc_pos / count

    def force_v_p(self):
        #Calculates the force vs position for all of the files in the data directory.
        #First check to make sure files are loaded and force vs position is computed.
        if type(self.fobjs) == str:
            self.load_dir(pos_loader)
        
        #self.load_dir(pos_loader)
        
    
    def avg_force_v_p(self, axis = 2, bin_size = 0.5, cant_indx = 24):
        #Averages force vs positon over files with the same potential. Returns a list of average force vs position for each cantilever potential in the directory.
        if type(self.fobjs) == str:
            self.load_dir(pos_loader)
        
        extractor = lambda fobj: [fobj.binned_cant_data[axis, 2], fobj.binned_pos_data[axis,2], fobj.electrode_settings[cant_indx]] #extracts [cant data, pos data, cant voltage]
        
        extracted = np.array(map(extractor, self.fobjs))
        self.ave_force_vs_pos = {}
        for v in np.unique(extracted[:, 2]):
            boolv = extracted[:, 2] == v
            xout, yout, yerrs = sbin(np.hstack(extracted[boolv, 0]), np.hstack(extracted[boolv, 1]), bin_size)
            self.ave_force_vs_pos[str(v)] =  [xout, yout, yerrs]

    def avg_pos_data(self):
        if type(self.fobjs) == str:
            self.load_dir(ft_loader)

        avg = self.fobjs[0].pos_data
        counts = 1.
        for obj in self.fobjs[1:]:
            for i in range(len(avg)):
                avg[i] += obj.pos_data[i]
            counts += 1.
        for i in range(len(avg)):
            avg[i] = avg[i] / counts
        self.ave_pos_data = avg

    def diagonalize_ave_pos(self):
        if type(self.Havg) == str:
            self.build_avgH()
        if type(self.ave_pos_data) == str:
            self.avg_pos_data()

        ft = np.fft.rfft(self.ave_pos_data)
        H = np.linalg.inv(self.Havg)
        
        ft_diag = np.einsum('ij, jk -> ik', H, ft)

        return np.fft.irfft(ft_diag)


    def H_vec(self, pcol = 1, ecol = 3):
        #Generates an array of Hs for the whole directory.
        #First check to make sure files are loaded and H is computed.
        if type(self.fobjs) == str: 
            self.load_dir(H_loader)
        
        if type(self.fobjs[0].H) == str:
            self.load_dir(H_loader)
            
        Her = lambda fobj: np.mean(fobj.H.Hmats[pcol, ecol, :], axis = 0)
        self.Hs = map(Her, self.fobjs)

    
    def build_uncalibrated_H(self):
        # Loop over file objects and construct a dictionary with frequencies 
        # as keys and 3x3 transfer matrices as values
        if type(self.fobjs) == str:
            self.load_dir(H_loader)

        Hout = {}
        Hout_noise = {}

        Hout_counts = {}

        for obj in self.fobjs:
            einds = obj.H.electrodes
            finds = obj.H.finds
            freqs = obj.fft_freqs[finds]
            
            for i in range(len(freqs)):
                if freqs[i] not in Hout:
                    Hout[freqs[i]] = np.zeros((3,3), dtype=np.complex128)
                    Hout_noise[freqs[i]] = np.zeros((3,3), dtype=np.complex128)
                    Hout_counts[freqs[i]] = np.zeros(3)

                outind = emap(einds[i])
                Hout[freqs[i]][:,outind] += obj.H.Hmats[:,einds[i],i]
                Hout_noise[freqs[i]][:,outind] += obj.noiseH.Hmats[:,einds[i],i]
                Hout_counts[freqs[i]][outind] += 1

        # Compute the average transfer function
        for key in Hout.keys():
            for i in [0,1,2]:
                Hout[key][:,i] = Hout[key][:,i] / Hout_counts[key][i]
                Hout_noise[key][:,i] = Hout_noise[key][:,i] / Hout_counts[key][i]
        
        self.Hs = Hout
        self.noiseHs = Hout_noise

    def save_H(self, fname):
        pickle.dump(self.Hs, open(fname, "wb"))

    def load_H(self, fname):
        newH = pickle.load( open(fname, "rb"))
        self.Hs = newH



    def calibrate_H(self):
        return





    def build_avgH(self, fthresh = 80):
        # average over frequencies f < 0.5*f_natural
        if type(self.Hs) == str:
            self.build_uncalibrated_H()

        keys = self.Hs.keys()

        mats = []
        for key in keys:
            if key < fthresh:
                mats.append(self.Hs[key])

        mats = np.array(mats)
        self.Havg =  np.mean(mats, axis=0)

        


    def plot_H(self, phase=False, show=True, label=False, noise=False,\
               show_zDC=False):
        # plot all the transfer functions

        if type(self.Hs) == str:
            print "need to build H's first..."
            self.build_uncalibrated_H()
            
        if noise:
            keys = self.noiseHs.keys()
        else:
            keys = self.Hs.keys()
        keys.sort()

        mats = []
        for freq in keys:
            if noise:
                mats.append(self.noiseHs[freq])
            else:
                mats.append(self.Hs[freq])

        # Plot the magnitude of the transfer function:
        #     Makes separate plots for a given direction of drive
        #     each with three subplots detailing x, y, and z response
        #     to a drive in a particular direction
        mats = np.array(mats)
        for drive in [0,1,2]:
            plt.figure(drive+1)
            for response in [0,1,2]:
                ax1 = plt.subplot(3,1,response+1)                    
                mag = np.abs(mats[:,response,drive])

                # check for NaNs from empty directory or incomplete
                # measurements and replace with unity
                nans = np.isnan(mag)
                mag[nans] = np.zeros(len(mag[nans])) + 1
                if np.mean(mag) == 0 and np.std(mag) == 0:
                    mag = mag + 1.

                if label and response == 0:
                    plt.loglog(keys, mag, label = self.label)
                elif show_zDC and response == 2:
                    plt.loglog(keys, mag, \
                               label="Avg Z: %0.4f"%self.ave_dc_pos[-1])
                else:
                    plt.loglog(keys, mag)

                if show:
                    ax1.legend(loc=0)
            plt.xlabel("Frequency [Hz]")
        
        for drive in [0,1,2]:
            plt.figure(drive+1)
            plt.subplot(3,1,1)
            plt.title("Drive in direction \'%i\'"%drive)

        # Plot the phase of the transfer function:
        #     Same plot/subplot breakdown as before
        if phase and not noise:
            for drive in [0,1,2]:
                plt.figure(drive+4)
                for response in [0,1,2]:
                    ax2 = plt.subplot(3,1,response+1)
                    phase = np.angle(mats[:,response,drive])

                    # Check for NaNs in phase and replace with ~0 
                    # (the semilogx doesn't like when one vector is 
                    # identically 0 so I add 1e-12)
                    nans = np.isnan(phase)
                    phase[nans] = np.zeros(len(phase[nans])) + 1e-12
                    if np.mean(phase) == 0 and np.std(phase) == 0:
                        phase = phase + 1e-12
                    unphase = np.unwrap(phase)
                    if unphase[0] < -2.5:
                        unphase = unphase + 2 * np.pi

                    if label and response == 0:
                        plt.semilogx(keys, unphase, label = self.label)
                    elif show_zDC and response == 2:
                        plt.semilogx(keys, unphase, \
                                   label="Avg Z: %0.4f"%self.ave_dc_pos[-1])
                    else:
                        plt.semilogx(keys, unphase)

                    if show:
                        ax2.legend(loc=0)

                plt.xlabel("Frequency [Hz]")

            for drive in [0,1,2]:
                plt.figure(drive+4)
                plt.subplot(3,1,1)
                plt.title("Drive in direction \'%i\'"%drive)

        # If the show command was on a noise plot, the phase plots
        # need to be correctly labeled with their legend, as the phase
        # response of the noise is never plotted and thus this function
        # never reaches the 'ax2.legend()' line in the phase block above
        elif noise and show:
            if show:
                for drive in [0,1,2]:
                    plt.figure(drive+4)
                    for response in [0,1,2]:
                        plt.subplot(3,1,response+1)
                        plt.legend(loc=0)
            
        # Show all the plots that have been built up
        if show:
            plt.show()

                

        
    def step_cal(self, dir_obj, n_phi = 140, plate_sep = 0.004, amp_gain = 200.):
        #Produce a conversion between voltage and force given a directory with single electron steps.
        #Check to see that Hs have been calculated.
        if type(dir_obj.Hs) == str:
            dir_obj.H_vec()
        
        phi = np.mean(np.angle(dir_obj.Hs[0:n_phi])) #measure the phase angle from the first n_phi samples.
        yfit =  np.abs(dir_obj.Hs)*np.cos(np.angle(dir_obj.Hs) - phi)
        plt.plot(yfit, 'o')
        plt.show()
        nstep = input("Enter guess at number of steps and charge at steps [[q1, q2, q3, ...], [x1, x2, x3, ...], vpq]: ")
        
        #function for fit with volts per charge as only arg.
        def ffun(x, vpq):
            qqs = vpq*np.array(nstep[0])
            return multi_step_fun(x, qqs, nstep[1])

        xfit = np.arange(len(dir_obj.Hs))
        
        #fit
        p0 = nstep[2]#Initial guess for the fit
        popt, pcov = curve_fit(ffun, xfit, yfit, p0 = p0)

        fitobj = Fit(popt, pcov, ffun)#Store fit in object.

        f, axarr = plt.subplots(2, sharex = True)#Plot fit
        fitobj.plt_fit(xfit, yfit, axarr[0])
        fitobj.plt_residuals(xfit, yfit, axarr[1])
        plt.show()
        
        #Determine force calibration.
        fitobj.popt *= 1./(amp_gain*bu.e_charge/plate_sep)
        fitobj.errs *= 1./(amp_gain*bu.e_charge/plate_sep)
        self.charge_step_calibration = fitobj
        
        
    def save_dir(self):
        #Method to save Data_dir object.
        if(not os.path.isdir(self.out_path) ):
            os.makedirs(self.out_path)
        outfile = os.path.join(self.out_path, "dir_obj.p")       
        pickle.dump(self, open(outfile, "wb"))

    def load_from_file(self):
        #Method to laod Data_dir object from a file.
        fname = os.path.join(self.out_path, "dir_obj.p")       
        temp_obj = pickle.load(open(fname, 'rb'))
        self.fobjs = temp_obj.fobjs
        self.Hs = temp_obj.Hs
        self.thermal_calibration = temp_obj.thermal_calibration
        self.charge_step_calibration = temp_obj.charge_step_calibration
        self.ave_force_vs_pos = temp_obj.ave_force_vs_pos

