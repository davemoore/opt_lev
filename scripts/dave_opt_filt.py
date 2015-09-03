## take a directory containing chirp data with charge and save the transfer function
## (i.e. complex response relative to the drive)

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = "/data/20140623/Bead1/one_charge_chirp"

chirp_low_freq = 10 # Hz
chirp_high_freq = 200 # Hz

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/dcmoore/analysis" + path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

def get_tf( fname ):

    print "Processing ", fname
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))
    fsamp = attribs["Fsamp"]    

    xdat = dat[:, bu.data_columns[0]]
    drive = dat[:, bu.drive_column] 

    wind = np.hanning( len(xdat) )

    ## first filter the response around the drive frequency to throw
    ## out noise
    b, a = sp.butter(5, [2.*(chirp_low_freq)/fsamp, 2.*(chirp_high_freq)/fsamp ], btype = 'bandpass')
    #xdat = sp.filtfilt( b, a, xdat )
    #drive = sp.filtfilt( b, a, drive ) 

    xf = np.fft.rfft( xdat*wind )
    df = np.fft.rfft( drive*wind )
    freq = np.fft.rfftfreq( len(xdat), 1./fsamp )

    H = xf/df

    if( False ):
        plt.figure()
        plt.semilogy(freq, np.abs(H) )

        #plt.figure()
        #plt.plot(freq, np.angle(H) )    

        plt.show()

    return H, freq
        
init_list = glob.glob(path + "/*.h5")
files = sorted(init_list, key = bu.find_str)

Htot = []
ntraces = 0
for f in files[:-5]:
    try:
        cH, cf = get_tf( f )

        if( len(Htot) > 0 ):
            Htot += cH
        else:
            Htot = cH
            ntraces +=1 
    except:
        print "sdjcbsdkjcbs"
Htot /= ntraces

plt.loglog(np.abs(Htot))
plt.show()
## now fit Lorentzian to phase and angle
#spars = [5e5, 145, 200, np.pi, 200.]
#fit_freq = [chirp_low_freq, chirp_high_freq]
#fit_points = bu.inrange( cf, fit_freq[0], fit_freq[1] )
#fit_points = np.logical_and( fit_points, np.logical_or(cf<100,np.abs(Htot) > 1e-2))
#bp, bcov = opt.curve_fit( bu.bead_spec_comp, cf[fit_points], Htot[fit_points], p0=spars )
#ffn = lambda x,p0,p1,p2,p3,p4: bu.bead_spec_comp(x,p0,p1,p2,p3)*bu.low_pass_tf(x,p4)
#bp, bcov = bu.curve_fit_complex( ffn, cf[fit_points], Htot[fit_points], spars)
#bp = spars

plt.figure()
plt.loglog( cf, np.abs(Htot), '.', markersize=1 )
plt.show()
#plt.plot( cf[fit_points], np.abs(ffn( cf[fit_points], bp[0], bp[1], bp[2], bp[3], bp[4] )), 'r' )
#plt.xlim([chirp_low_freq, chirp_high_freq])

#cang = np.angle(Htot)
#cang[cang < 0] += 2*np.pi


plt.figure()
plt.semilogx(cf, np.angle(Htot), '.', markersize=1 )
#plt.plot( cf[fit_points], np.angle(ffn( cf[fit_points], bp[0], bp[1], bp[2], bp[3], bp[4] )), 'r' )
plt.xlim([chirp_low_freq, chirp_high_freq])
plt.show()

Hout = ffn(cf,bp[0], bp[1], bp[2], bp[3], bp[4])

def plot_resp( fname ):
    print "Processing ", fname
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))
    fsamp = attribs["Fsamp"]    

    xdat = dat[:, bu.data_columns[0]]
    drive = dat[:, bu.drive_column] 

    wind = np.hanning( len(xdat) )
    df = np.fft.rfft( drive )
    freq = np.fft.rfftfreq( len(xdat), 1./fsamp )

    drive_pred = np.fft.irfft( df*Hout )

    plt.figure()
    plt.plot( xdat )
    plt.plot( drive_pred)
    plt.show()


# ## now apply the transfer function to get the expected response in the time domain
# for f in files[:-1]:
#     plot_resp( f )

np.save(os.path.join(outpath, "trans_func.npy"), Hout )

