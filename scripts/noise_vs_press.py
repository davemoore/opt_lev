import numpy as np
import bead_util as bu
import matplotlib
import os
import glob


path = "/data/20150324/Bead1/freq_sweep_chirp"
reprocessfile = True

def ave_power(psd, freqs, fmin, fmax):
    ##returns the average power in psd within a specified frequency range.
    return np.sum(psd[(freqs>=fmin) & (freqs<=fmax)])


if reprocessfile:
    init_list = glob.glob(path + "/*.h5") 

def get_data
