
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
import opt_filt_util as ofu
from scipy.interpolate import UnivariateSpline

path = "/data/20140711/Bead6/cal_charge_chirps_100s"
path_drive = "/data_slave/20140711/Bead6/cal_charge_chirps_100s"
H_path = "/home/arider/analysis/20140711/Bead6/one_charge_chirp"
noise_path = "/data/20140623/Bead1/chirp_plates_terminated"

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/arider/analysis" + path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

reprocessfile = True
plot_angle = False
plot_phase = False
remove_laser_noise = False
remove_outliers = True
plot_flashes = False
plt_fit = False

ref_file = 0 ## index of file to calculate angle and phase for
file_start = 0
N_pts = 500000
drive_column = -1
fsamp = 5000.


of = open(os.path.join(outpath, "H_dict.pkl"), "rb")
H_dict = pickle.load(of)
of.close()


noise = np.load("/home/arider/analysis/20140623/Bead1/plates_terminated/noise.npy")


all_freqs = np.fft.rfftfreq(N_pts, 1./fsamp)
bfreqs = np.array([i in H_dict['freqs'] for i in all_freqs])


b1 = all_freqs>20.
b2 = all_freqs<190.
bt = -b1-b2


def getdata(fname, path):
	print "Processing ", fname
        dat, attribs = bu.getdata(os.path.join(path, fname))
        #dat_drive, cf_drive = bu.getdata_drive(os.path.join(path_drive, fname_drive))
        
        ## make sure file opened correctly
        if( len(dat) == 0 ):
            return {}

        ctime = attribs["time"]

        dfft = np.fft.rfft(dat[:, drive_column])
        rfft = np.fft.rfft(dat[:, 0])
        
        ps = H_dict["phase_spline"]
        Gs = H_dict["G_spline"]
        H = np.exp(1.j*ps(all_freqs))*Gs(all_freqs)

        Ahat, sig_Ahat, rcs = ofu.opt_filter(dfft[bt], rfft[bt], noise[bt], H[bt])
        
        if plt_fit:
            fig = plt.figure()
            plt.subplot(3, 1, 1)
            plt.semilogy(freqs[b1], np.abs(np.real(rfft[b1])), 'b.', label = 'signal')
            plt.semilogy(freqs[b1], np.abs(np.real(Ahat*H[b1]*dfft[b1])), 'r.', label = 'fit')
            plt.ylabel('Real')
            plt.subplot(3, 1, 2)
            plt.semilogy(freqs[b1], np.abs(np.imag(rfft[b1])), 'b.', label = 'signal')
            plt.semilogy(freqs[b1], np.abs(np.imag(Ahat*H[b1]*dfft[b1])), 'r.', label = 'fit')
            plt.ylabel('Complex')
            plt.subplot(3, 1, 3)
            plt.plot(freqs[b1], np.abs(H[b1]*Ahat*dfft[b1]-rfft[b1])**2/noise[b1], 'r.', label = 'Contribution to Chi-square')
            plt.legend()
            plt.ylabel('Residuals')
            plt.xlabel("Frequency[Hz]")
            plt.show()
            
        ## make a dictionary containing the various calculations
        out_dict = {"Ahat": np.real(Ahat),
                    "sig_Ahat": sig_Ahat,
                    "rcs": rcs,
                    "temps": attribs["temps"],
                    "time": bu.labview_time_to_datetime(ctime),
                    "num_flashes": attribs["num_flashes"]}

        
        
        return out_dict

    
if reprocessfile:

    init_list = glob.glob(path + "/*.h5")
    files = sorted(init_list, key = bu.find_str)
    files = files[::1]

    corrs_dict = {}
    for f in files[file_start:]:
        curr_dict = getdata(f, path)

        for k in curr_dict.keys():
            if k in corrs_dict:
                corrs_dict[k].append( curr_dict[k] )
            else:
                corrs_dict[k] = [curr_dict[k],]
    
    of = open(os.path.join(outpath, "processed.pkl"), "wb")
    pickle.dump( corrs_dict, of )
    of.close()
else:
    of = open(os.path.join(outpath, "processed.pkl"), "rb")
    corrs_dict = pickle.load( of )
    of.close()


## first plot the variation versus time
dates = matplotlib.dates.date2num(corrs_dict["time"])
As = np.array(corrs_dict["Ahat"])*2.
rcs = np.array(corrs_dict["rcs"])
temp1 = np.array(corrs_dict["temps"])[:,0]
temp2 = np.array(corrs_dict["temps"])[:,1]
num_flashes = np.array(corrs_dict["num_flashes"])

scale_fac = ofu.scale_fac(As)
print "scale_fac:", scale_fac

b2 = dates>=dates[0]

#print As
print "Mean is", np.mean(As), "Median is", np.median(As)
print np.std(As)/np.sqrt(len(As)-1)



plt.figure() 
plt.plot_date(dates, As*scale_fac.x, 'b.', label="Max corr")
plt.plot_date(dates, rcs, 'r.', label = "RCS")
    

plt.xlabel("Time")
plt.ylabel("Optimal charge")


fig1 = plt.figure() 
plt.subplot(1,2,1)
resid_data = As-np.round(As)
dates = np.array(dates)[b2]
resid_data = np.array(resid_data)[b2]

plt.plot_date(dates, resid_data, 'r.', markersize=2, label="A")
ax = plt.gca()
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

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

if( remove_outliers ):

    ## throw out any bad times before doing the fit
    time_window = 5 ## mins
    nsig = 5
    bad_points = np.argwhere(np.abs(resid_data > bp[1]+nsig*bp[2]))
    pts_to_use = np.logical_not(bad_points)

    print np.sum(pts_to_use)
    
    plt.plot_date(dates, resid_data, 'k.', markersize=2, label="Max corr")
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
plt.show()

def make_corr_plot( amp_vec, corr_vec, col, lab=""):
    ## get a list of the drive amplitudes
    drive_list = amp_vec
    amp_list = np.transpose(np.vstack((corr_vec, np.zeros_like(corr_vec))))

    sf = 1.0 ##np.median( amp_list[:,0] )
    #plt.plot( amp_vec, corr_vec/sf, '.', color=[col[0]+0.5, col[1]+0.5, col[2]+0.5], zorder=1)
    plt.errorbar( drive_list, amp_list[:,0]/sf, yerr=amp_list[:,1]/sf, fmt='.', color=col, linewidth = 1.5, label=lab )
    fit_pts = drive_list < 40
    p = np.polyfit( drive_list[fit_pts], amp_list[fit_pts,0]/sf, 1)
    xx = np.linspace( np.min(drive_list), np.max(drive_list), 1e2)
    plt.plot(xx, np.polyval(p, xx), color=col, linewidth = 1.5)
    #plt.xlim([0, 1e3])
    plt.xlabel("Drive voltage [V]")
    plt.ylabel("Correlation with drive signal [V]")
    plt.legend(loc="upper left", numpoints = 1)
    #plt.ylim([-1, 2])


