## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, re
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = r"D:\Data\20140529\Bead1\charge_with_ref"
reprocessfile = True
plot_angle = False
ref_file = 0 ## index of file to calculate angle and phase for

scale_fac = 1.
scale_file = 1.

fsamp = 5000.
fdrive = 41
fref = 1027
NFFT = 2**12
phaselen = int(fsamp/fdrive) #number of samples used to find phase
plot_scale = 1. ## scaling of corr coeff to units of electrons
plot_offset = 1.
data_columns = [0, 1] ## column to calculate the correlation against
drive_column = 3

b, a = sp.butter(3, [2.*(fdrive-5)/fsamp, 2.*(fdrive+5)/fsamp ], btype = 'bandpass')
boff, aoff = sp.butter(3, 2.*(fdrive-10)/fsamp, btype = 'lowpass')

def rotate_data(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c*x - s*y, s*x + c*y

def getangle(fname):
        print "Getting angle from: ", fname 
        num_angs = 100
        dat, attribs = bu.getdata(os.path.join(path, fname))
        pow_arr = np.zeros((num_angs,2))
        ang_list = np.linspace(-np.pi/2.0, np.pi/2.0, num_angs)
        for i,ang in enumerate(ang_list):
            rot_x, rot_y = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
            pow_arr[i, :] = [np.std(rot_x), np.std(rot_y)]
        
        best_ang = ang_list[ np.argmax(pow_arr[:,0]) ]
        print "Best angle [deg]: %f" % (best_ang*180/np.pi)

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
        dat, attribs = bu.getdata(os.path.join(path, fname))
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        #xdat = sp.filtfilt(b, a, xdat)
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
        maxv = np.argmax(corr2) 
        print maxv
        return maxv


def getdata(fname, maxv, ang):

	print "Processing ", fname
        dat, attribs = bu.getdata(os.path.join(path, fname))

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        lentrace = len(xdat)
        ## zero pad one cycle
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr_full = np.correlate( xdat, dat[:,drive_column])/lentrace
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
                    "time": bu.labview_time_to_datetime(ctime)}

        return out_dict


if reprocessfile:
  init_list = os.listdir(path)
  if( 'processed.pkl' in init_list):
    bad_idx = init_list.index( 'processed.pkl' )
    del init_list[bad_idx]
  files = sorted(init_list, key = lambda str:int(re.findall('\d+', str)[-2]))

  ang = getangle(files[ref_file])
  phase = getphase(files[ref_file], ang)
  corrs_dict = {}
  for f in files:
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

plt.figure() 
plt.plot_date(dates, corr_t0/np.median(corr_t0), 'b.', label="Corr at t=0")
plt.plot_date(dates, max_corr/np.median(max_corr), 'r.', label="Max corr")
plt.plot_date(dates, psd/np.median(psd), 'k.', label="PSD")
plt.plot_date(dates, ref_psd/np.median(ref_psd), '.', color=[0.5, 0.5, 0.5], label="Ref. PSD")
plt.plot_date(dates, temp1/np.median(temp1), 'g', label="Laser temp")
plt.plot_date(dates, temp2/np.median(temp2), 'c', label="Amp temp")
plt.xlabel("Time")
plt.ylabel("Correlation with drive")
plt.legend(numpoints = 1)


plt.show()

