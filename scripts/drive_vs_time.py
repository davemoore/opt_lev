## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = "/data/20140617/Bead3/ramp_overnight"
reprocessfile = True
plot_angle = False
ref_file = 0 ## index of file to calculate angle and phase for

file_start = 0

scale_fac = 1./0.0017
scale_file = 1.

amp_gain = 200.

fsamp = 5000.
fdrive = 41.
fref = 1027
NFFT = 2**14
phaselen = int(fsamp/fdrive) #number of samples used to find phase
plot_scale = 1. ## scaling of corr coeff to units of electrons
plot_offset = 1.
data_columns = [0, 1] ## column to calculate the correlation against
drive_column = 3

b, a = sp.butter(3, [2.*(fdrive-1)/fsamp, 2.*(fdrive+1)/fsamp ], btype = 'bandpass')
boff, aoff = sp.butter(3, 2.*(fdrive-10)/fsamp, btype = 'lowpass')

def rotate_data(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c*x - s*y, s*x + c*y

def getangle(fname):
        print "Getting angle from: ", fname 
        num_angs = 100
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        pow_arr = np.zeros((num_angs,2))
        ang_list = np.linspace(-np.pi/2.0, np.pi/2.0, num_angs)
        for i,ang in enumerate(ang_list):
            rot_x, rot_y = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
            pow_arr[i, :] = [np.std(rot_x), np.std(rot_y)]
        
        best_ang = ang_list[ np.argmax(pow_arr[:,0]) ]
        print "Best angle [deg]: %f" % (best_ang*180/np.pi)

        cf.close()

        if(plot_angle):
            plt.figure()
            plt.plot(ang_list, pow_arr[:,0], label='x')
            plt.plot(ang_list, pow_arr[:,1], label='y')
            plt.xlabel("Rotation angle")
            plt.ylabel("RMS at drive freq.")
            plt.legend()
            
            ## also plot rotated time stream
            rot_x, rot_y = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], best_ang)
            plt.figure()
            plt.plot(rot_x)
            plt.plot(rot_y)
            plt.plot(dat[:, drive_column] * np.max(rot_x)/np.max(dat[:,drive_column]))
            plt.show()
        
        

        return best_ang

def getphase(fname, ang):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        #xdat = sp.filtfilt(b, a, xdat)
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
        maxv = np.argmax(corr2) 

        cf.close()

        print maxv
        return maxv


def getdata(fname, maxv, ang):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        dat[:, drive_column] *= amp_gain
        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]
        
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        dat[:, drive_column] = sp.filtfilt(b, a, dat[:, drive_column])
        ydat =  sp.filtfilt(b, a, ydat)
        lentrace = len(xdat)
        ## zero pad one cycle
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        drive_amp = np.sqrt(2)*np.std( dat[:,drive_column] )
        corr_full = np.correlate( xdat, dat[:,drive_column])/(lentrace*drive_amp**2)*scale_fac
        corr = corr_full[ maxv ]
        corr_max = np.max(corr_full)
        corr_max_pos = np.argmax(corr_full)
        xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 
        #ypsd, freqs = matplotlib.mlab.psd(ydat, Fs = fsamp, NFFT = NFFT) 
        max_bin = np.argmin( np.abs( freqs - fdrive ) )
        ref_bin = np.argmin( np.abs( freqs - fref ) )

        xoff = sp.filtfilt(boff, aoff, xdat)

        if(False):
            plt.figure()
            plt.plot( xdat )
            plt.plot( dat[:, drive_column] )

            plt.figure()
            plt.plot( corr_full )
            plt.show()

        ctime = attribs["time"]

        curr_scale = 1.0
        ## make a dictionary containing the various calculations
        out_dict = {"corr_t0": corr,
                    "max_corr": [corr_max, corr_max_pos],
                    "psd": np.sqrt(xpsd[max_bin]),
                    "ref_psd": np.sqrt(xpsd[ref_bin]),
                    "temps": attribs["temps"],
                    "time": bu.labview_time_to_datetime(ctime),
                    "num_flashes": attribs["num_flashes"]}

        cf.close()
        return out_dict

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
    

if reprocessfile:
  init_list = os.listdir(path)
  if( 'processed.pkl' in init_list):
    bad_idx = init_list.index( 'processed.pkl' )
    del init_list[bad_idx]
  if( 'current_corr.txt' in init_list):
    bad_idx = init_list.index( 'current_corr.txt' )
    del init_list[bad_idx]
  files = sorted(init_list, key = find_str)

  ang = 0 ##getangle(files[ref_file])
  phase = getphase(files[ref_file], ang)
  corrs_dict = {}
  for f in files[file_start:-1]:
    curr_dict = getdata(f, phase, ang)

    for k in curr_dict.keys():
        if k in corrs_dict:
            corrs_dict[k].append( curr_dict[k] )
        else:
            corrs_dict[k] = [curr_dict[k],]
    
  of = open("processed.pkl", "wb")
  pickle.dump( corrs_dict, of )
  of.close()
else:
  of = open("processed.pkl", "rb")
  corrs_dict = pickle.load( of )
  of.close()

## first plot the variation versus time
dates = matplotlib.dates.date2num(corrs_dict["time"])
corr_t0 = np.array(corrs_dict["corr_t0"])
max_corr = np.array(corrs_dict["max_corr"])[:,0]
psd = np.array(corrs_dict["psd"])
ref_psd = np.array(corrs_dict["ref_psd"])
temp1 = np.array(corrs_dict["temps"])[:,0]
temp2 = np.array(corrs_dict["temps"])[:,1]
num_flashes = np.array(corrs_dict["num_flashes"])

plt.figure() 
plt.plot_date(dates, corr_t0, 'r.', label="Max corr")

## fit a polynomial to the ref pdf
p = np.polyfit(dates, ref_psd/np.median(ref_psd), 1)
xx = np.linspace(dates[0], dates[-1], 1e3)


def plot_avg_for_per(x, y, idx1, idx2, linecol):
    ## get the average and error (given by std of points) for a sub period between flashes
    eval, eerr = np.median(y[idx1:idx2]), np.std(y[idx1:idx2])/np.sqrt(idx2-idx1)
    ax = plt.gca()

    mid_idx = int( (idx1 + idx2)/2 )
    ax.vlines(x[mid_idx],eval-eerr,eval+eerr, color=linecol, linewidth=1.5)
    hash_width = (x[idx2]-x[idx1])/10.
    ax.hlines(eval+eerr,x[mid_idx]-hash_width,x[mid_idx]+hash_width, color=linecol, linewidth=1.5)
    ax.hlines(eval-eerr,x[mid_idx]-hash_width,x[mid_idx]+hash_width, color=linecol, linewidth=1.5)

    return x[mid_idx], eval
    

flash_idx = np.argwhere( num_flashes > 0 )

yy = plt.ylim()
## plot the location of the flashes and average each period between
## make sure to plot for first period
avg_vals = []
if(len(flash_idx)>1):
    plot_avg_for_per( dates, corr_t0, 0, flash_idx[0], 'r')
    for i,f in enumerate(flash_idx):
        plt.plot_date( [dates[f], dates[f]], yy, 'k--')
        if( i < len(flash_idx)-1 ):
            cx, eval_corr = plot_avg_for_per( dates, corr_t0, flash_idx[i], flash_idx[i+1], 'r')
            eval_psd = 0.0
            avg_vals.append( [cx, eval_corr, eval_psd] )

plt.ylim(yy)

plt.xlabel("Time")
plt.ylabel("Correlation with drive")
plt.legend(numpoints = 1, loc="upper left")


fig1 = plt.figure() 
plt.subplot(1,2,1)
plt.plot_date(dates, corr_t0-np.round(corr_t0), 'r.', markersize=2, label="Max corr")
## set limits at +/- 5 sigma
cmu, cstd = np.median(corr_t0-np.round(corr_t0)), np.std(corr_t0-np.round(corr_t0))
yy = plt.ylim([cmu-5*cstd, cmu+5*cstd])
for i,f in enumerate(flash_idx):
    plt.plot_date( [dates[f], dates[f]], yy, marker=None, linestyle='-', color=[0.5, 0.5, 0.5])
plt.plot_date(dates, corr_t0-np.round(corr_t0), 'r.', markersize=2, label="Max corr")
plt.ylim(yy)
plt.xlabel("Time")
plt.ylabel("Residual to nearest integer charge [$e$]")
ax = plt.gca()
plt.subplot(1,2,2)
ax2 = plt.gca()
ax2.yaxis.set_visible(False)
hh, be = np.histogram( corr_t0-np.round(corr_t0), bins = 30, range=yy )
bc = be[:-1]+np.diff(be)/2.0

## fit the data
def gauss_fun(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )
ax.set_position(matplotlib.transforms.Bbox(np.array([[0.125,0.1],[0.675,0.9]])))
ax2.set_position(matplotlib.transforms.Bbox(np.array([[0.725,0.1],[0.9,0.9]])))

bp, bcov = opt.curve_fit( gauss_fun, bc, hh, p0=[10, 0, 0.03] )

xx = np.linspace(yy[0], yy[1], 1e3)
plt.errorbar( hh, bc, xerr=np.sqrt(hh), yerr=0, fmt='k.', linewidth=1.5 )
plt.plot( gauss_fun(xx, bp[0], bp[1], bp[2]), xx, 'r', linewidth=1.5, label="$\mu$ = %.3e $\pm$ %.3e $e$"%(bp[1], np.sqrt(bcov[1,1])))
plt.legend()
plt.ylim(yy)

plt.xlabel("Counts")

#diff plot of mean values
#avg_vals = np.array(avg_vals)
#if(len(avg_vals)>2):
#    plt.figure()
#    hh, be = np.histogram( np.diff( avg_vals[:,1] ) )
#    plt.step(be[:-1], hh, where='post')


# ## now do the diff plot
# plt.figure() 
# plt.plot_date(dates[:-1], np.diff(corr_t0/np.median(corr_t0)), 'r.', label="Max corr")
# plt.plot_date(dates[:-1], np.diff(psd/np.median(psd)), 'k.', label="PSD")
# yy = plt.ylim()
# for f in flash_idx:
#     plt.plot_date( [dates[f-1], dates[f-1]], yy, 'k--')
# plt.ylim(yy)

# plt.xlabel("Time")
# plt.ylabel("Diff from last point")
# plt.legend(numpoints = 1, loc="upper right")


# ## now histo up the ones after a flash and those with no flash
# corr_diff = np.diff( (corr_t0/np.median(corr_t0)) ) ## / (np.polyval(p, dates)))
# flash_locs = np.zeros( len(corr_diff) ) > 0
# for f in flash_idx:
#   flash_locs[f-1] = True

# corr_diff_noflash = corr_diff[np.logical_not(flash_locs)]
# corr_diff_flash = corr_diff[flash_locs]

# hh_noflash, be = np.histogram(corr_diff_noflash, bins=50, range=[-0.1, 0.1], normed=False)
# hh_flash, be = np.histogram(corr_diff_flash, bins=50, range=[-0.1, 0.1], normed=False)

# plt.figure()
# plt.step(be[:-1], hh_noflash, 'b', label="No flash")
# plt.step(be[:-1], hh_flash, 'r', label="Flash")
# plt.legend(numpoints=1)

plt.show()

