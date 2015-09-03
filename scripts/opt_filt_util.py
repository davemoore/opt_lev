##Set of utility functions for performing an optimal filter of bead data

import numpy as np
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle
import opt_filt_util as ofu
import sys
import matplotlib



def align_templates(temp1, temp2, samps = 100):
    #returns the number of samples of offset between temp1 and temp2. looks over samps offsets.
    temp1 = np.append(np.zeros(samps/2), np.append(temp1, np.zeros(samps/2)))
    corrs = np.zeros((samps/2)*2+1)
    len_trace = len(temp2)
    for i in range(len(corrs)):
        corrs[i] = np.sum(temp1[i:i+len_trace]*temp2) 

    #print np.max(corrs)/np.sum(temp1**2)
    return np.argmax(corrs) - samps/2

def getdata_fft(fname, temp_path, drive_column = -1, response_column = 0):
#Stripped stripped down data loader for gettin frequency domain data fro drive and response columns.

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(temp_path, fname), other_comp=True)

        ## make sure file opened correctly
        if( len(dat) == 0 ):
            print 'shit'
            return {} 

        fsamp = attribs["Fsamp"]
        volt_div = attribs["volt_div"]
        

        #dpsd, freqs = matplotlib.mlab.psd(dat[:, drive_column], 2**16, Fs = fsamp)

        #plt.loglog(freqs, dpsd)
        #plt.show()

        drive_fft = np.fft.rfft(dat[:, drive_column])
        response_fft = np.fft.rfft(dat[:, response_column])
        N = len(dat[:, response_column])
        cf.close()
        return {"drive_fft": drive_fft, "response_fft": response_fft, "fsamp": fsamp, "N":N}

def get_trans_func(path, drive_column = -1, response_column = 0, fl = 10, fu = 200, plot_H = False):
    #given a path, averages over all of the drive and response ffts and divides them to estimate the transfer function.

    init_list = glob.glob(path + "/*.h5")
    files = sorted(init_list, key = bu.find_str)
    init_dict  = getdata_fft(files[0], path)
    H_sum = init_dict['response_fft']/init_dict['drive_fft']

    for f in files[1:5]:
        try:
            curr_dict = getdata_fft(f, path)
            H_sum+=curr_dict['response_fft']/curr_dict['drive_fft']
        
        except:
            print "Holy Shit:",sys.exc_info()[0]
            files.remove(f)

    H_ave = H_sum/len(files)

    freqs = np.fft.rfftfreq(len(H_ave), d = 1/init_dict['fsamp'])

    b1 = freqs>fl
    b2 = freqs<fu
    
    
    
    if plot_H:
        plt.loglog(freqs[-b1-b2], np.abs(H_ave)[-b1-b2])
        plt.show()

    return freqs[-b1-b2], H_ave[-b1-b2]


def get_noise(path):
    #Returns the amplitude squared of the noise in each bin averaged over all files in path.
    init_list = glob.glob(path + "/*.h5")
    files = sorted(init_list, key = bu.find_str)
    N_files = len(files)
    print "there are", N_files
    N_bad = 0
    curr_dict = ofu.getdata_fft(files[0], path)
    noise_pow = np.abs(curr_dict['response_fft'])**2
    for i in range(N_files)[1:]:
        
        try:
            curr_dict = ofu.getdata_fft(files[i], path)
            noise_pow += np.abs(curr_dict['response_fft'])**2
              

        
        
        except:
            print "Holy Shit:",sys.exc_info()[0]                   
            N_bad +=1

    return noise_pow/(N_files-N_bad)

    
def freq_fit(A, H, drive_fft, response_fft, noise_pow):
    #returns chi-squared score for the amplitude A
    return np.sum((np.abs(response_fft-A*H*drive_fft))**2/noise_pow)

def opt_filter(drive_fft, response_fft, noise_psd,trans_func):

    temp = trans_func*drive_fft

    Ahat =  np.sum(np.conjugate(temp)*response_fft/noise_psd)/np.sum(np.abs(temp)**2/noise_psd)
    
    sig_Ahat = np.sqrt(2*np.sum(np.abs(temp)**2)/noise_psd)**-1

    rcs = np.sum(np.abs(response_fft-Ahat*temp)**2/noise_psd)/(len(drive_fft)-2)
    
    return Ahat, sig_Ahat, rcs

def scale_fun(A, charges):
    return np.sum((A*charges-np.round(A*charges))**2)

def scale_fac(charges):
    scale_fun = lambda A: np.sum((charges/A-np.round(charges/A))**2)
    res = opt.minimize_scalar(scale_fun, bounds = (.8, 2.), method = 'bounded')#, tol = 0.00001)
    return res


def opt_filter2(response_fft, temp,  noise_psd):


    Ahat =  np.sum(np.conjugate(temp)*response_fft/noise_psd)/np.sum(np.abs(temp)**2/noise_psd)
    
    sig_Ahat = np.sqrt(2*np.sum(np.abs(temp)**2)/noise_psd)**-1

    rcs = np.sum(np.abs(response_fft-Ahat*temp)**2/noise_psd)/(len(response_fft)-2)
    
    return Ahat, sig_Ahat, rcs

