
## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = "/data/20140811/Bead4/chargelp_cal"

## path to directory containing charge steps, used to calibrate phase and 
## scaling.  leave empty to use data path
cal_path ="/data/20140807/Bead1/chargelp_cal"

noise_path = "/data/20140807/Bead1/plates_terminated"



## path to save plots and processed files (make it if it doesn't exist)
out_path = "/home/arider/analysis" + path[5:]
if( not os.path.isdir( out_path ) ):
    os.makedirs(out_path)

reprocessfile = True
get_noise = True
get_template = True
file_start = 0


fsamp = 5000.
data_columns = [0, 1] ## column to calculate the correlation against
drive_column = -1

if reprocessfile:
    if get_noise:
        noise = bu.get_noisepsd_path(noise_path)
        np.save(os.path.join(out_path, 'noise.npy'), noise)
    else:
        noise = np.load(os.path.join(out_path, 'noise.npy'))

    if get_template:
        temp = bu.get_template(cal_path)
        np.save(os.path.join(out_path, 'temp.npy'), temp)
    else:
        temp = np.load(os.path.join(out_path, 'temp.npy'))

    amps_dict = bu.amp_opt_filter_path(path, temp, noise[:, 0], Allfiles = False)
    #freqs = np.fft.rfftfreq(500000, 1./5000.)
    #thetas, phis = bu.get_forceangle_of(path, noise, temp)
    #thetas_cal, phis_cal = bu.get_forceangle_of(cal_path, noise, temp)
    #amps_dict['thetas'] = thetas-np.median(thetas_cal)
    #amps_dict['phis'] = phis - np.median(phis_cal)

    of = open(os.path.join(out_path, "processed.pkl"), "wb")
    pickle.dump(amps_dict, of)
    of.close()

else:
    of = open(os.path.join(out_path, "processed.pkl"), "rb")
    amps_dict = pickle.load(of)
    
    of.close()


## first plot the variation versus time
print amps_dict.keys()
dates = matplotlib.dates.date2num(map(bu.labview_time_to_datetime, amps_dict["time"]))
amplitudes = np.real(np.array(amps_dict['amplitudes']))
temp1 = np.array(amps_dict["temps"])[:,0]
temp2 = np.array(amps_dict["temps"])[:,1]
num_flashes = np.array(amps_dict["num_flashes"])
drive_amp = np.array(amps_dict["drive_amplitude"])
#thetas = np.array(amps_dict['thetas'])
#phis = np.array(amps_dict['phis'])

plt.figure() 
plt.plot(dates, amplitudes)
plt.xlabel('time')
plt.ylabel('charge[e]')
plt.show()

plt.figure()
#plt.plot(range(len(thetas)), thetas, 'xr', label = 'thetas')
#plt.plot(range(len(phis)), phis, 'ob', label = 'phis')
plt.legend()
plt.show()


fig1 = plt.figure() 
plt.subplot(1,2,1)

resid_data = amplitudes
plt.plot_date(dates, resid_data, 'r.', markersize=2, label="amplitude")
## set limits at +/- 5 sigma
cmu, cstd = np.median(resid_data), np.std(resid_data)
yy = plt.ylim([cmu-5*cstd, cmu+5*cstd])
plt.ylim(yy)
plt.xlabel("Time")
plt.ylabel("Residual to nearest integer charge [$e$]")
ax = plt.gca()

hh, be = np.histogram( resid_data, bins = np.max([30, len(resid_data)/50]), range=yy )
bc = be[:-1]+np.diff(be)/2.0

## fit the data
def gauss_fun(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )

amp0 = np.sum(hh)/np.sqrt(2*np.pi*cstd)
bp, bcov = opt.curve_fit( gauss_fun, bc, hh, p0=[amp0, cmu, cstd] )

if(False):

    ## throw out any bad times before doing the fit
    

    plt.plot_date(dates, resid_data, 'k.', markersize=2, label="residual amplitude")
    cmu, cstd = np.median(resid_data), np.std(resid_data)
    hh, be = np.histogram( resid_data, bins = np.max([50, len(resid_data)/50]), range=[cmu-10*cstd, cmu+10*cstd] )
    bc = be[:-1]+np.diff(be)/2.0
    amp0 = np.sum(hh)/np.sqrt(2*np.pi*cstd)
    bp, bcov = opt.curve_fit( gauss_fun, bc, hh, p0=[amp0, cmu, cstd] )

plt.subplot(1,2,2)
ax2 = plt.gca()
ax2.yaxis.set_visible(False)
ax.set_position(matplotlib.transforms.Bbox(np.array([[0.125,0.1],[0.675,0.9]])))
ax2.set_position(matplotlib.transforms.Bbox(np.array([[0.725,0.1],[0.9,0.9]])))



xx = np.linspace(yy[0], yy[1], 1e3)
plt.errorbar( hh, bc, xerr=np.sqrt(hh), yerr=0, fmt='k.', linewidth=1.5 )
plt.plot( gauss_fun(xx, bp[0], bp[1], bp[2]), xx, 'r', linewidth=1.5, label="$\mu$ = %.3e $\pm$ %.3e $e$"%(bp[1], np.sqrt(bcov[1,1])))
plt.legend()
plt.ylim(yy)

plt.xlabel("Counts")

## plot correlation with drive squared vs voltage
def make_corr_plot( amp_vec, corr_vec, col, lab=""):
    ## get a list of the drive amplitudes
    drive_list = amp_vec
    amp_list = np.transpose(np.vstack((corr_vec, np.zeros_like(corr_vec))))

    sf = 1.0 ##np.median( amp_list[:,0] )
    #plt.plot( amp_vec, corr_vec/sf, '.', color=[col[0]+0.5, col[1]+0.5, col[2]+0.5], zorder=1)
    plt.errorbar( drive_list, amp_list[:,0]/sf, yerr=amp_list[:,1]/sf, fmt='.', color=col, linewidth = 1.5, label=lab )
    fit_pts = drive_list < 40000.
    p = np.polyfit( drive_list[fit_pts], amp_list[fit_pts,0]/sf, 1)
    xx = np.linspace( np.min(drive_list), np.max(drive_list), 1e2)
    plt.plot(xx, np.polyval(p, xx), color=col, linewidth = 1.5)
    #plt.xlim([0, 1e3])
    plt.xlabel("Drive voltage [V]")
    plt.ylabel("Correlation with drive signal [V]")
    plt.legend(loc="upper left", numpoints = 1)
    #plt.ylim([-1, 2])

plt.show()


