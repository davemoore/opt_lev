import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit

data_dir1 = r"C:\Data\20160429\beam_profiles1"
data_dir2 = r"C:\Data\20160429\beam_profiles_b10"
data_dir3 = r"C:\Data\20160429\beam_profiles_b20"
data_dir4 = r"C:\Data\20160429\beam_profiles_b30"
data_dir5 = r"C:\Data\20160429\beam_profiles_u10"
data_dir6 = r"C:\Data\20160429\beam_profiles_u20"
data_dir7 = r"C:\Data\20160429\beam_profiles_u30"
data_dir7 = r"C:\Data\20160429\beam_profiles_u44"
data_dir8 = r"C:\Data\20160429\beam_profiles_sideside"
data_dir9 = r"C:\Data\20160429\beam_profiles_sideside_half"

data_dir10 = r"C:\Data\20160621\no_bead\beam_profile_init"
data_dir11 = r"C:\Data\20160621\no_bead\beam_profile_b35"
data_dir12 = r"C:\Data\20160621\no_bead\beam_profile_b80"
data_dir13 = r"C:\Data\20160621\no_bead\beam_profile_init2"
data_dir14 = r"C:\Data\20160621\no_bead\beam_profile_u40"
data_dir15 = r"C:\Data\20160621\no_bead\beam_profile_u75"

data_dir16 = r"C:\Data\20160621\no_bead\beam_profile_fine_init"
data_dir17 = r"C:\Data\20160621\no_bead\beam_profile_fine_u20"
data_dir18 = r"C:\Data\20160621\no_bead\beam_profile_fine_u40"

data_dir19 = r"C:\Data\20160711\nobead\beam_profile2"
data_dir20 = r"C:\Data\20160711\nobead\beam_profile_slow"

data_dir21 = r"C:\Data\20160923\beam_profiles\z0"
data_dir22 = r"C:\Data\20160923\beam_profiles\z10"
data_dir23 = r"C:\Data\20160923\beam_profiles\z20"
data_dir24 = r"C:\Data\20160923\beam_profiles\z30"
data_dir25 = r"C:\Data\20160923\beam_profiles\z40"
data_dir26 = r"C:\Data\20160923\beam_profiles\z50"
data_dir27 = r"C:\Data\20160923\beam_profiles\z60"
data_dir28 = r"C:\Data\20160923\beam_profiles\z70"
data_dir29 = r"C:\Data\20160923\beam_profiles\z80"

data_dir30 = r"C:\Data\20160923\beam_profiles\z0_2"
data_dir31 = r"C:\Data\20160923\beam_profiles\z0_3"



#data_dirs = [data_dir19, data_dir20]
#data_dirs = [data_dir16, data_dir17, data_dir18]
#labels = ['2016-7-14', '2016-7-14 - Slow']
#labels = ['init', 'u20', 'u40']

#data_dirs = [data_dir21, data_dir22, data_dir23, data_dir24, data_dir25, \
#             data_dir26, data_dir27, data_dir28, data_dir29]
#labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80']

data_dirs = [data_dir21, data_dir30, data_dir31]
labels = ['old', 'new', 'new new']

#stage x = col 17, stage y = 18, stage z = 19



def spatial_bin(xvec, yvec, bin_size = .5):
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals/=fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    for i, b in enumerate(bins):
        idx = bins_vals == b
        y_binned[i] =  np.mean(yvec[idx])
        y_errors[i] = scipy.stats.sem(yvec[idx])
    return np.array(bins), np.array(y_binned), np.array(y_errors)
    
        
    

def profile(fname, ends = 5000, stage_cal = 8., data_column = 6, stage_column = 19):
    dat, attribs, f = bu.getdata(fname)
    dat = dat[ends:-ends, :]
    dat[:, stage_column]*=stage_cal
    #plt.plot(dat[:, data_column])
    #plt.show()
    f.close()
    b, a = sig.butter(3, 0.25)
    int_filt = sig.filtfilt(b, a, dat[:, data_column])    
    stage_filt = sig.filtfilt(b, a, dat[:, stage_column])
    v = np.gradient(stage_filt)
    b = np.abs(v) > 1e-4
    b2 = np.gradient(stage_filt)>0
    proft = np.gradient(int_filt[b])/v[b]
    #dir_sign = np.sign(np.gradient(stage_filt))
    b, y, e = spatial_bin(dat[:, stage_column][b], proft)
    return b, y, e


fitfun = lambda x, s, a, m: a*np.exp(-(x-m)**2/(2*s**2))

def proc_dir(dir, data_column = 6, stage_column = 19,fit = False):
    files = glob.glob(dir + '\*.h5')
    #print files
    bs = np.array([])
    ys = np.array([])
    for f in files:
        b, y, e = profile(f, data_column = data_column, stage_column = stage_column)
    
        bs = np.append(bs, b)
        ys = np.append(ys, y)

    bi, y, e =  spatial_bin(bs, ys)
    #y /= np.max(y)
    y = np.abs(y)
    binx = np.argmax(y)
    cent = b[binx]
    plt.errorbar(b -cent ,y , e , fmt = '.', label = labels[i])
    plt.yscale('log')
    if fit:
        p0 = [2,0.4,0] # sigma, A, mean
        fitpts = np.abs(b - cent) < 4
        popt, pcov = curve_fit(fitfun, b[fitpts]-cent, y[fitpts], p0 = p0, maxfev = 10000)
        plt.plot(b-cent, fitfun(b-cent, *popt), 'r', linewidth  = 2, label = "Gaussian fit sigma = %f $\mu$m"%popt[0])

for i,f in enumerate(data_dirs):
    print f
    proc_dir(f, fit=False)#True)
               

#plt.gca().set_yscale('')
plt.xlabel('Position [um]')
plt.ylabel('Intensity [arbitrary units]')
#plt.yscale('log')
#plt.ylim([.0, 1.5])
plt.legend()
plt.show()


