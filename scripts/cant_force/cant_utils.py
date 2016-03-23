import numpy as np
import matplotlib
import bead_util as bu
import scipy
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Define functions and classes for use processing and fitting data.
def thermal_psd_spec(f, A, f0, g):
    #The position power spectrum of a microsphere normalized so that A = (volts/meter)^2*2kb*t/M
    w = 2.*np.pi*f #Convert to angular frequency.
    w0 = 2.*np.pi*f0
    num = g
    denom = ((w0**2 - w**2)**2 + w**2*g**2)
    return A*num/denom

class fit:
    #holds the optimal parameters and errors from a fit
    def __init__self(self, popt, pcov):
        self.popt = popt
        self.errs = np.diagonal(pcov)

def thermal_fit(psd, freqs, fit_freqs = [10., 300.], kelvin = 300., fudge_fact = 1e-6):
    #Function to fit the thermal spectra of a bead's motion
    #First need good intitial guesses for fit parameters.
    fit_bool = bu.inrange(freqs, fit_freqs[0], fit_freqs[1]) #Boolian vector of frequencies over which the fit is performed
    f0 = freqs[np.argmax(psd[fit_bool])] #guess resonant frequency from hightest part of spectrum
    df = freqs[1] - freqs[0] #Frequency increment.
    vpmsq = bu.bead_mass/(bu.kb*kelvin)*np.sum(psd[fit_bool])*df*len(psd)/np.sum(fit_bool) #Guess at volts per meter using equipartition
    g0 = 1./2.*f0 #Guess at damping assuming critical damping
    A0 = vpmsq*2.*bu.kb*kelvin/(bu.bead_mass*fudge_fact)
    p0 = [A0, f0, g0] #Initial parameter vectors 
    popt, pcov = curve_fit(thermal_psd_spec, freqs[fit_bool], psd[fit_bool], p0 = p0)
    return fit(popt, pcov)


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
    

#define a class with all of the attributes and methods necessary for processing a single data file to 
    

class data:
    #This is a class with all of the attributes and methods for a single data file.

    def __init__(self):
        self.fname = "Filename not assigned."
        #self.path = "Directory not assigned." #Assuming directory in filename
        self.pos_data = "bead position data not loaded"
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

    def load(self, fstr, sep, cant_cal = 8., stage_travel = 80.):
        #Methods to load the attributes from a single data file. sep is a vector of the distances of closes approach for each direction ie. [xsep, ysep, zsep] 
        dat, attribs, f = bu.getdata(fstr)
        self.fname = fstr

        #Data vectors and their transforms
        self.pos_data = np.transpose(dat[:, 0:3]) #x, y, z bead position
        self.cant_data = np.transpose(np.resize(sep, np.shape(np.transpose(self.pos_data)))) + stage_travel - np.transpose(dat[:, 17:20])*cant_cal
        self.electrode_data = dat[:, 8:16] #Record of voltages on the electrodes
        
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
        f.close()


    def spatial_bin(self, bin_size = 0.5, cant_axis = 2):
        #Method for spatially binning data based on stage z  position.
        
        self.binned_cant_data = [[], [], []]
        self.binned_pos_data = [[], [], []]
        self.binned_data_errors = [[], [], []]

        for i, v in enumerate(self.pos_data): 
            bins, y_binned, y_errors = sbin(self.cant_data[cant_axis], v, bin_size)
            self.binned_cant_data[i] = bins
            self.binned_pos_data[i] = y_binned 
            self.binned_data_errors[i] = y_errors 
        
        self.binned_cant_data = np.array(self.binned_cant_data)
        self.binned_pos_data = np.array(self.binned_pos_data)
        self.binned_data_errors = np.array(self.binned_data_errors)

    def psd(self, NFFT = 2**15):
        #uses matplotlib mlab psd to take a psd of the microsphere position data.
        #Need to preallocate memory for psds
        self.psds = [[], [], []]
    
        for i, v in enumerate(self.pos_data):
            psd, freqs = matplotlib.mlab.psd(v, NFFT = NFFT, Fs = self.Fsamp)
            
            self.psds[i] = np.transpose(psd)[0]
        
        self.psds = np.array(self.psds)
        self.psd_freqs = freqs

    def fft(self):
        #Uses numpy fft rfft to compute the fft of the position data
        self.fft = np.fft.rfft(self.pos_data)
        self.fft_freqs = np.fft.rfftfreq(np.shape(self.pos_data)[1])*self.Fsamp


    def thermal_calibration(self, calf, make_plot = False):
        #Use thermal calibration calibrate voltage scale into physical units
        #Check to see if psds is computed and compute if not.
        if type(self.psds) == str:
            self.psd()
            
        self.thermal_cal = []
        for i, v in enumerate(self.psds):
            p
            
#Define a class to hold information about a whole directory of files.
