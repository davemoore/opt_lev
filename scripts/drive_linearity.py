## loads files containing the position response versus voltage and
## plots the linearity of the response

import numpy as np
import matplotlib, math, os, re
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.optimize as opt

dirlist = ["Bead1/Vramp", "Bead1/Vramp_day2", "Bead1/Vramp_trial3", 
           "Bead1/Vramp_trial4", "Bead1/Vramp_trial5",
           "Bead1/Vramp_trial6", "Bead1/Vramp_trial7", "Bead1/Vramp_trial8", "Bead1/Vramp_trial9"]
#path = "Bead1/Vramp_day2"
cal_file = "Bead1/2mbar_axcool.dat"
damp_file = "Bead1/2e-5mbar_xyzcool.dat"
reprocessfile = True
plot_angle = False
ref_file = -1 ## index of file to calculate angle and phase for

scale_fac = 1.
scale_file = 1.

fsamp = 1500.
fdrive = 30
NFFT = 2**10
phaselen = int(fsamp/fdrive) #number of samples used to find phase
plot_scale = 1. ## scaling of corr coeff to units of electrons
plot_offset = 1.
data_columns = [0, 1] ## column to calculate the correlation against
drive_column = 3

b, a = sp.butter(3, [2.*(fdrive-5)/fsamp, 2.*(fdrive+5)/fsamp ], btype = 'bandpass')
boff, aoff = sp.butter(3, 2.*(fdrive-20)/fsamp, btype = 'lowpass')

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


def getdata(fname, maxv, ang, make_plot=False):
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
        corr_full = np.correlate( xdat, dat[:,drive_column])/(lentrace*np.std(dat[:,drive_column]))
        corr = corr_full[ maxv ]
        corr_max = np.max(corr_full)
        corr_max_pos = np.argmax(corr_full)
        xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 
        #ypsd, freqs = matplotlib.mlab.psd(ydat, Fs = fsamp, NFFT = NFFT) 
        max_bin = np.argmin( np.abs( freqs - fdrive ) )

        if(make_plot):
            plt.figure(spec_fig.number)
            plt.loglog( freqs, np.sqrt(xpsd) )

        curr_scale = 1.0
        return [corr, corr_max, corr_max_pos, np.std(xoff), np.sqrt(xpsd[max_bin]), ang, curr_volt(fname)/2000.]

def func2(f, A, f0, Damping):
    omega = 2*math.pi*f
    omega_0 = 2*math.pi*f0
    return np.sqrt(A*Damping/((omega_0**2 - omega**2)**2 + omega**2*Damping**2))


def abs_cal( cf, fit_points = [0, NFFT/2], skip_points = False ):

    dat = np.loadtxt(cf, skiprows = 5, usecols = [2, 3, 4, 5] )
    xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)

    cpsd_x, f = mlab.psd( xdat, Fs=fsamp, NFFT=NFFT )
    cpsd_y, f = mlab.psd( ydat, Fs=fsamp, NFFT=NFFT )


    ##first, fit for the absolute calibration
    spars = [1, 80, 2000]

    if(skip_points):
        xdat_fit = f[fit_points[0]:fit_points[1]]
        ydat_fit = np.sqrt(cpsd_x[fit_points[0]:fit_points[1]])
        bad_pts = ydat_fit > ydat_fit[0] * 1.5
        xdat_fit = xdat_fit[ np.logical_not(bad_pts) ]
        ydat_fit = ydat_fit[ np.logical_not(bad_pts) ]
        func3 = lambda x, A, f0, gam: func2(x, A, res_freq, gam)
        bp, bcov = opt.curve_fit( func3, xdat_fit, ydat_fit, p0=spars)
    else:
        xdat_fit = f[fit_points[0]:fit_points[1]]
        ydat_fit = np.sqrt(cpsd_x[fit_points[0]:fit_points[1]])
        bp, bcov = opt.curve_fit( func2, xdat_fit, ydat_fit, p0=spars)



    print bp

    norm_rat = (2*1.38e-23*300/(0.1e-12)) * 1/bp[0]

    plt.figure(44)
    plt.loglog( f, np.sqrt(norm_rat * cpsd_x), '.' )
    plt.loglog( xdat_fit, np.sqrt(norm_rat * ydat_fit**2), 'k.' )
    xx = np.linspace( f[fit_points[0]], f[fit_points[1]], 1e3)
    plt.loglog( xx, np.sqrt(norm_rat * func2( xx, bp[0], bp[1], bp[2] )**2), 'r')
    plt.xlabel("Freq [Hz]")
    plt.ylabel("PSD [m Hz$^{-1/2}$]")
    plt.show()
    
    return np.sqrt(norm_rat), bp[1], bp[2]

curr_volt = lambda str:int(re.findall('\d+', str)[-1])



## get angle from first directory
path = dirlist[0]
init_list = os.listdir(path)
if( 'processed.txt' in init_list):
    bad_idx = init_list.index( 'processed.txt' )
    del init_list[bad_idx]
files = sorted(init_list, key = curr_volt)
ang = getangle(files[ref_file])

cal_fac, res_freq, axdamp = abs_cal( cal_file )

damp_fit = abs_cal( damp_file, [10, 150], True )

spec_fig = plt.figure()
fit_fig = plt.figure()
baseline_fig = plt.figure()
for path in dirlist:

    if reprocessfile:
      init_list = os.listdir(path)
      if( 'processed.txt' in init_list):
        bad_idx = init_list.index( 'processed.txt' )
        del init_list[bad_idx]
      files = sorted(init_list, key = curr_volt)

      ang = getangle(files[ref_file])
      phase = getphase(files[ref_file], ang)
      corrs = []

      for f in files:
        #curr_ang = getangle(f)
        curr_ang = ang
        if( f == files[-1] ):
            make_plot = True
        else:
            make_plot = False
        corrs.append(getdata(f, phase, curr_ang, make_plot))
      #getdata2 = lambda f: getdata(f, phase, ang) 
      #corrs = np.array(map(getdata2, files))
      corrs = np.array(corrs)
      np.save('processed.npy', corrs)
    else:
        corrs = np.load('processed.npy')

    ## fit max correlation
    def ffn( x, A ):
        return A*x

    bp, bcov = opt.curve_fit( ffn, corrs[:,6], cal_fac*corrs[:,1], p0=[corrs[-1,1]/corrs[-1,6]] )

    xx = np.linspace( 0, corrs[-1,6], 1e2)

    plt.figure(fit_fig.number)
    plt.plot(corrs[:,6],cal_fac*corrs[:,1], '.')
    plt.plot( xx, ffn(xx, bp) , label="A = %.3e $\pm$ %.3e" % (bp, bcov[0,0]))

    #plt.figure(baseline_fig.number)
    #plt.plot(corrs[:,6],cal_fac*corrs[:,3], '.')

    plt.xlabel("Voltage (V)")
    plt.ylabel("Correlated motion (m)")
    plt.legend(numpoints = 1, loc="upper left")

    ## now num charges
    zm = np.sqrt( damp_fit[2]**2 + ((2*np.pi*res_freq)**2 - (2*np.pi*fdrive)**2)**2 )
    ##print "Num charges: ", 0.1e-12 * (2*np.pi*fdrive) * zm * 2e-3 * bp[0] / 1.6e-19 
    print "Num charges: ", 0.1e-12*(2*np.pi*72)**2 * 2e-3 * bp[0]/ 1.6e-19


plt.show()

