## measure the force from the cantilever
import glob, os
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.mlab

data_dir = "/data/20141212/Bead2/charge_cant_vs_freq"

pos_list = ["-inf", "-5_5", "-5_0", "-4_5", "-4_0", "-3_5", "-3_0", "-2_5", "-2_0", "-1_5", "-1_0"]
pos_in = [-15, -5.5, -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1.]
flist = [10, 39, 68, 97, 126, 155, 184, 213, 242, 271]
press_list = [47, 2]

## make color list same length as flist
jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=len(flist))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

def qfun( d, p ):
    return 1.0*p/d**2
def cfun( d, p ):
    return 1.0*p/d**3

for p in press_list:

    plt.figure(p)

    for i,f in enumerate(flist):
        
        curr_dat = []
        for pin, pstr in zip(pos_in, pos_list):

            curr_file = os.path.join(data_dir, "%smbar_cant_%s_500mV_%dHz.h5"%(p,pstr,f))
            if( not os.path.isfile(curr_file) ):
                #print "Warning, couldn't find: ", curr_file
                continue
                
            cdat,attr,_ = bu.getdata( curr_file )

            xdat, ydat, zdat, ddat = cdat[:,bu.data_columns[0]], cdat[:,bu.data_columns[1]], cdat[:,bu.data_columns[2]], cdat[:,bu.drive_column]

            ## now find correlation with drive signal and drive squared
            corr_full_x = bu.corr_func(ddat, xdat, attr['Fsamp'], f)
            corr_full_y = bu.corr_func(ddat, ydat, attr['Fsamp'], f)
            corr_full_z = bu.corr_func(ddat, zdat, attr['Fsamp'], f)

            corr_full_x2 = bu.corr_func(ddat**2, xdat, attr['Fsamp'], f)
            corr_full_y2 = bu.corr_func(ddat**2, ydat, attr['Fsamp'], f)
            corr_full_z2 = bu.corr_func(ddat**2, zdat, attr['Fsamp'], f)        

            xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = attr['Fsamp'], NFFT = 2**17) 
            ypsd, freqs = matplotlib.mlab.psd(ydat, Fs = attr['Fsamp'], NFFT = 2**17) 
            zpsd, freqs = matplotlib.mlab.psd(zdat, Fs = attr['Fsamp'], NFFT = 2**17) 

            #cdist = np.sqrt(pin**2 + 1.5**2 + 1.75**2)/(0.125)*5
            cdist = np.sqrt((pin+0.75)**2 )/(0.125)*5
            out_vec = [cdist,]
            for c in [corr_full_x, corr_full_y, corr_full_z,
                      corr_full_x2, corr_full_y2, corr_full_z2]:
                #max_pos = np.argmax( np.abs( c ) )
                #out_vec.append(c[max_pos])
                out_vec.append( np.max(np.abs(c)) )

            for spec in [xpsd, ypsd, zpsd]:
                fund_bin = np.argmin(np.abs(freqs - f))
                harm_bin = np.argmin(np.abs(freqs - 2*f))
                out_vec.append(np.sqrt(spec[fund_bin]))
                out_vec.append(np.sqrt(spec[harm_bin]))

            curr_dat.append( out_vec )

        curr_dat = np.array(curr_dat)
        cidx = 11
        plt.plot( curr_dat[1:,0], curr_dat[1:,cidx]-curr_dat[1,cidx],'o',color=scalarMap.to_rgba(i), label='f=%dHz'%f )
        
        px2,_ = sp.curve_fit(qfun, curr_dat[1:,0], curr_dat[1:,cidx]-curr_dat[1,cidx], p0=-1e4)
        px3,_ = sp.curve_fit(cfun, curr_dat[1:,0], curr_dat[1:,cidx]-curr_dat[1,cidx], p0=-1e6)

        xx = np.linspace( np.min(curr_dat[1:,0]), np.max(curr_dat[1:,0]), 1e2)
        plt.plot(xx, qfun(xx, px2), ':', color=scalarMap.to_rgba(i), linewidth=1.5)
        plt.plot(xx, cfun(xx, px3), '-', color=scalarMap.to_rgba(i), linewidth=1.5)

        #plt.plot( curr_dat[1:,0], curr_dat[1:,12]-curr_dat[1,12],'s',markeredgecolor=scalarMap.to_rgba(i), markerfacecolor=[1,1,1], markeredgewidth=1.5 )
    
    ax = plt.gca()
    ax.set_yscale("log")

    #plt.legend(numpoints=1,loc=")
    plt.show()

