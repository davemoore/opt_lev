import matplotlib, os, h5py, glob
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np
import bead_util as bu

path = "/data/20140717/Bead5/reduce_feedback"

#file_list = glob.glob( os.path.join(path, "*.h5") )
file_list = ["urmbar_xyzcool_xgain_1_ygain_1_50mV_40Hz.h5",
             "urmbar_xyzcool_xgain_0_5_ygain_1_50mV_40Hz.h5",
             "urmbar_xyzcool_xgain_0_2_ygain_1_50mV_40Hz.h5",
             "urmbar_xyzcool_xgain_0_ygain_0_1_50mV_40Hz.h5"]

print file_list


col_list = ['k', [0, 0, 0.75]]
m_list = ['s', 'o']
damp_list = [10e-3, 500]

## first calibrate into physical units
ref_2mbar = "/data/20140717/Bead5/_50mV_1000Hz2mbar_zcool_100s.h5"
abs_cal, fit_bp, fit_cov = bu.get_calibration(ref_2mbar, [1,200],
                                              make_plot=True,
                                              NFFT=2**14,
                                              exclude_peaks=False)
print fit_bp

fig = plt.figure()
alist = []
for i,f in enumerate([file_list[-1], file_list[0]]):


    dat, attribs, cf = bu.getdata( os.path.join(path,f) )

    fsamp = attribs["Fsamp"]

    xdat = dat[:, 0]
    xdat -= np.median(xdat)

    if( len(xdat) == 500000):
        NFFT = 2**20
    else:
        NFFT = 2**14

    xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 

    xpsd = np.sqrt( xpsd ) * abs_cal * 1e9

    if(i == 0):
        fit_points = [120,143.]
        exc = [[126, 129],
               [134, 138]]
    else:
        fit_points = [50,150.]
        exc = False

    print i
    plt.figure()
    abs_calc, fit_bpc, fit_covc = bu.get_calibration(os.path.join(path,f), fit_points,
                                                  make_plot=False,
                                                  NFFT=NFFT,
                                                     exclude_peaks=exc, 
                                                     spars=[fit_bp[0]*10, 131.888, damp_list[i]])    
    alist.append([ fit_bpc[0], np.sqrt(fit_covc[0,0]) ])
    print "Fit pars:"
    print fit_bpc, fit_covc
    
    xx = np.linspace(fit_points[0], fit_points[1], 1e3)
    plt.figure( fig.number )
    if( i == 0 ):
        plt.semilogy(freqs[::20], xpsd[::20], markerfacecolor='None', markeredgecolor=col_list[i], marker=m_list[i], linestyle='None', markeredgewidth=1.5, markersize=4)
        plt.plot( xx, bu.bead_spec_rt_hz( xx, fit_bpc[0], fit_bpc[1], fit_bpc[2])*abs_cal*1e9, linewidth=3, color='g' )
    else:
        plt.semilogy(freqs, xpsd, markerfacecolor='None', markeredgecolor=col_list[i], marker=m_list[i], linestyle='None', markeredgewidth=1.5, markersize=4)
        plt.plot( xx, bu.bead_spec_rt_hz( xx, fit_bpc[0], fit_bpc[1], fit_bpc[2])*abs_cal*1e9, linewidth=3, color='r', linestyle='--' )


    plt.xlim([50, 150])
    plt.ylim([1e-1, 5e3])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Radial position PSD [nm Hz$^{-1/2}$]")


fig.set_size_inches(6,4)
plt.subplots_adjust(top=0.99, right=0.99, bottom=0.12)
plt.savefig("feedback_cooling.eps")

print "Effective temp:"
T0 = 297.4
alist = np.array(alist)
print alist[1,0]/alist[0,0]*T0, " +/- ", alist[1,1]/alist[0,0]*T0


plt.show()
