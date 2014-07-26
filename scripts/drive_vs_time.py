
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

path = "/data/20140724/Bead3/no_charge_chirp"
## path to directory containing charge steps, used to calibrate phase and 
## scaling.  leave empty to use data path
cal_path = "/data/20140724/Bead3/chargelp_fine_calib"
single_charge_fnums = [13, 14]

ref_2mbar = "/data/20140724/Bead3/2mbar_zcool_50mV_40Hz.h5"

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/dcmoore/analysis" + path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

reprocessfile = False
plot_angle = False
plot_phase = False
remove_laser_noise = False
remove_outliers = True
plot_flashes = False
ref_file = 0 ## index of file to calculate angle and phase for

file_start = 0

scale_fac = 3.
scale_file = 1.

## These gains should always be left as one as long as
## the voltage_div setting was set correctly when taking data
## Otherwise, they are the ratio of the true gain to the gain
## that was set
amp_gain = 1. ## gain to use for files in path
amp_gain_cal = 1.  ## gain to use for files in cal_path

fsamp = 5000.
fdrive = 41.
fref = 1027
NFFT = 2**14
phaselen = int(fsamp/fdrive) #number of samples used to find phase
plot_scale = 1. ## scaling of corr coeff to units of electrons
plot_offset = 1.
laser_column = 3

def getdata(fname, gain, resp_fit, resp_dat, orth_pars):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        ## make sure file opened correctly
        if( len(dat) == 0 ):
            return {}

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]
            drive_amplitude = attribs["drive_amplitude"]

        ## now get the drive recorded by the other computer (if available)
        fname_drive = fname.replace("/data", "/data_slave")
        fname_drive = fname_drive.replace(".h5", "_drive.h5")            

        ## gain is not set in the drive file, so use the one from the data file
        if( os.path.isfile( fname_drive ) ):
            drive_dat, drive_attribs, drive_cf = bu.getdata(fname_drive, gain_error=attribs['volt_div']*gain)
        else:
            drive_dat = None

        ## is this a calibration file?
        cdir,_ = os.path.split(fname)
        is_cal = cdir == cal_path

        ## now insert the drive column from the drive file (ignore for calibrations)
        if( not is_cal and drive_dat != None):
            dat[:,-1] = drive_dat[:,-1]

        drive_amp,_ = bu.get_drive_amp( dat[:,bu.drive_column], fsamp )


        ## now double check that the rescaled drive amp seems reasonable
        ## and warn the user if not
        curr_gain = bu.gain_fac( attribs['volt_div']*gain )
        offset_frac = np.abs( drive_amp/(curr_gain * attribs['drive_amplitude']/1e3 )-1.0)
        if( curr_gain != 1.0 and offset_frac > 0.1):
            print "Warning, voltage_div setting doesn't appear to match the expected gain for ", fname
            print "Skipping this point"
            return {}

        xdat, ydat, zdat = dat[:,bu.data_columns[0]], dat[:,bu.data_columns[1]], dat[:,bu.data_columns[2]]

        if( orth_pars ):
            xdat, ydat, zdat = bu.orthogonalize( xdat, ydat, zdat,
                                                 orth_pars[0], orth_pars[1], orth_pars[2] )

        ## make correlation in time domain with predicted drive (fixed to known phase)
        corr_fit = bu.corr_func(np.fft.irfft(resp_fit), xdat, fsamp, fsamp)[0]/drive_amp
        corr_dat = bu.corr_func(np.fft.irfft(resp_dat), xdat, fsamp, fsamp)[0]/drive_amp

        ## calculate optimal filter
        vt = np.fft.rfft( xdat )
        st = resp_fit
        of_fit = np.real(np.sum( np.conj(st) * vt / J)/np.sum( np.abs(st)**2/J ))/drive_amp

        st = resp_dat
        of_dat = np.real(np.sum( np.conj(st) * vt / J)/np.sum( np.abs(st)**2/J ))/drive_amp

        ctime = attribs["time"]

        ## make a dictionary containing the various calculations
        out_dict = {"corr": [corr_fit, corr_dat],
                    "of": [of_fit, of_dat],
                    "temps": attribs["temps"],
                    "time": bu.labview_time_to_datetime(ctime),
                    "num_flashes": attribs["num_flashes"],
                    "is_cal": is_cal,
                    "drive_amp": drive_amp}

        cf.close()
        return out_dict

if reprocessfile:

  init_list = glob.glob(path + "/*.h5")
  files = sorted(init_list, key = bu.find_str)

  if(cal_path):
      cal_list = glob.glob(cal_path + "/*.h5")
      cal_files = sorted( cal_list, key = bu.find_str )
      files = zip(cal_files[:-1],np.zeros(len(cal_files[:-1]))+amp_gain_cal) \
              + zip(files[:-1],np.zeros(len(files[:-1]))+amp_gain)

      ## make the transfer function from the calibration files with one
      ## charge
      tf_fit, tf_dat, orth_pars = bu.get_avg_trans_func( cal_files, single_charge_fnums )
      ## get the noise from the 0 charge files at 10V
      J = bu.get_avg_noise( cal_files, single_charge_fnums[1]+1, orth_pars, make_plot = False )

  else:
      print "Warning, no calibration path defined.  Assuming default response function"
      
      files = zip(files[:-1],np.zeros(len(files[:-1]))+amp_gain)      
      
  corrs_dict = {}
  for f,gain in files[file_start:]:
    curr_dict = getdata(f, gain, tf_fit, tf_dat, orth_pars)

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

## apply rough calibration to the various metrics by 
## assuming ~1 charge on the calibration points
is_cal = np.array(corrs_dict["is_cal"])

sfac_rel = [ 1./np.median(np.array(corrs_dict["corr"])[is_cal,0]),
             1./np.median(np.array(corrs_dict["corr"])[is_cal,1]), 
             1./np.median(np.array(corrs_dict["of"])[is_cal,0]), 
             1./np.median(np.array(corrs_dict["of"])[is_cal,1]) ]

dates = matplotlib.dates.date2num(corrs_dict["time"])
corr_fit = np.array(corrs_dict["corr"])[:,0]*scale_fac*sfac_rel[0]
corr_dat = np.array(corrs_dict["corr"])[:,1]*scale_fac*sfac_rel[1]
of_fit = np.array(corrs_dict["of"])[:,0]*scale_fac*sfac_rel[2]
of_dat = np.array(corrs_dict["of"])[:,1]*scale_fac*sfac_rel[3]
temp1 = np.array(corrs_dict["temps"])[:,0]
temp2 = np.array(corrs_dict["temps"])[:,1]
num_flashes = np.array(corrs_dict["num_flashes"])
drive_amp = np.array(corrs_dict["drive_amp"])

fig = plt.figure() 
plt.plot_date(dates, corr_fit, 'r.', label="Corr from fit")
plt.plot_date(dates, corr_dat, 'g.', label="Corr from dat")
plt.plot_date(dates, of_fit, 'k.', label="OF from fit")
plt.plot_date(dates, of_dat, 'b.', label="OF from dat")
plt.legend(numpoints=1)
plt.xlabel("Time")
plt.ylabel("Correlation with drive [e]")
plt.title("Comparison of correlation calculations")

amp_for_plotting = of_fit

## now do absolute calibration as well
if(ref_2mbar):
    abs_cal, fit_bp, fit_cov = bu.get_calibration(ref_2mbar, [1,200],
                                                  make_plot=True,
                                                  NFFT=2**14,
                                                  exclude_peaks=False)
    scale_fac_abs = (bu.bead_mass*(2*np.pi*fit_bp[1])**2)*bu.plate_sep/(bu.e_charge) * abs_cal
    corr_abs = (amp_for_plotting)/scale_fac*scale_fac_abs
    fig2 = plt.figure()
    plt.plot(dates, amp_for_plotting, 'r.', label="Step calibration")
    plt.plot(dates, corr_abs, 'k.', label="Absolute calibration")

    plt.xlabel("Time")
    plt.ylabel("Correlation with drive [e]")
    plt.title("Comparison of calibrations")


    plt.show()


    




fig1 = plt.figure() 
plt.subplot(1,2,1)

resid_data = amp_for_plotting-np.round(amp_for_plotting)
plt.plot_date(dates, resid_data, 'r.', markersize=2, label="Max corr")
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
    pts_to_use = np.logical_not(is_cal)
    #pts_to_use = np.logical_and(np.logical_not(is_cal), bu.inrange(drive_amp, 5, 2000))
    print np.sum(pts_to_use)
    for p in bad_points:
        pts_to_use[ np.abs(dates - dates[p]) < time_window/(24.*60.)] = False

    plt.plot_date(dates[pts_to_use], resid_data[pts_to_use], 'k.', markersize=2, label="Max corr")
    cmu, cstd = np.median(resid_data[pts_to_use]), np.std(resid_data[pts_to_use])
    hh, be = np.histogram( resid_data[pts_to_use], bins = np.max([50, len(resid_data[pts_to_use])/50]), range=[cmu-10*cstd, cmu+10*cstd] )
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

