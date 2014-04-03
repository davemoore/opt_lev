## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import scipy.signal as sp
import scipy.optimize as opt

path = "Bead1/27Hz100mV"
reprocessfile = False
plot_angle = False
ref_file = 0 ## index of file to calculate angle and phase for

scale_fac = 1.
scale_file = 1.

fsamp = 1500.
fdrive = 27
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
	dat = np.loadtxt(os.path.join(path, fname), skiprows = 5, usecols = [2, 3, 4, 5] )
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
	dat = np.loadtxt(os.path.join(path, fname), skiprows = 5, usecols = [2, 3, 4, 5] )
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        #xdat = sp.filtfilt(b, a, xdat)
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
        maxv = np.argmax(corr2) 
        print maxv
        return maxv


def getdata(fname, maxv, ang):
	print "Processing ", fname
        dat = np.loadtxt(os.path.join(path, fname), skiprows = 5, usecols = [2, 3, 4, 5] )
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        xdat = sp.filtfilt(b, a, xdat)
        xoff = sp.filtfilt(boff, aoff, xdat)
        #corr_full = np.correlate(xdat[:phaselen],dat[:phaselen,drive_column], 'full')
        #corr_full = np.correlate(xdat,dat[:,drive_column], 'full')
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

        if(False):
            plt.figure()
            plt.plot( xdat )
            plt.plot( dat[:, drive_column] )

            plt.figure()
            plt.plot( corr_full )
            plt.show()

        curr_scale = 1.0
        return [corr, corr_max, corr_max_pos, np.std(xoff), np.sqrt(xpsd[max_bin]), ang]


if reprocessfile:
  init_list = os.listdir(path)
  if( 'processed.txt' in init_list):
    bad_idx = init_list.index( 'processed.txt' )
    del init_list[bad_idx]
  files = sorted(init_list, key = lambda str:int(re.findall('\d+', str)[2]))

  ang = getangle(files[ref_file])
  phase = getphase(files[ref_file], ang)
  corrs = []

  for f in files:
    #curr_ang = getangle(f)
    curr_ang = ang
    corrs.append(getdata(f, phase, curr_ang))
  plt.show()
  #getdata2 = lambda f: getdata(f, phase, ang) 
  #corrs = np.array(map(getdata2, files))
  corrs = np.array(corrs)
  np.save('processed.npy', corrs)
else:
    corrs = np.load('processed.npy')


plt.figure() 
plt.plot(corrs[:,0]/np.median(corrs[:,0]), 'c.', label="Corr at t=0")
plt.plot(corrs[:,1]/np.median(corrs[:,1]), 'r.', label="Max corr")
plt.plot(corrs[:,3]/np.median(corrs[:,3]), 'g.', label="Xoff rms")
plt.plot(corrs[:,4]/np.median(corrs[:,4]), 'b.', label="PSD")
normed_resp = corrs[:,1]/corrs[:,3]
plt.plot(normed_resp/np.median(normed_resp), 'k.', label="Normed")
plt.xlabel("File number")
plt.ylabel("Correlation with drive")
plt.legend(numpoints = 1)

if( len(corrs[:,0]) > 10 ):
    h, be = np.histogram(corrs[:,5], bins=len(corrs[:,5])/5.)
    plt.figure()
    plt.step(be[:-1], h, 'k', where='post')

plt.show()

