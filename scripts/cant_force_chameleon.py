## measure the force from the cantilever, averaging over files
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab
import matplotlib.dates as mdates

#data_dir = "/data/20150908/Bead2/cant_mod"
data_dir = "/data/20150928/backgrounds/basepressure_12Hz_2"
remake_files = True


NFFT = 2**18

#mod_freq = 3.05
mod_freq = 13

lower_plot_width = 10
upper_plot_width = 10

#conv_fac = 1.6e-15/0.11 * (1./0.1) # N to V, assume 10 
conv_fac = 4.29e-13 #* (1./0.2) # N to V 

def sort_fun( s ):
    return float(re.findall("\d+.h5", s)[0][:-3])

flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
print flist

tempdata, tempattribs, temphandle = bu.getdata(flist[0])
drive_freq = tempattribs['stage_settings'][-2]

temphandle.close()

if(remake_files):
    tot_dat = []
    tot_psd = []
    npsd = 0
    for f in flist[::]:

        print f 

        cpos = float(re.findall("Z\d+nm", f)[0][1:-2])
        #drive_freq = float(re.findall("\d+Hz",f)[0][:-2])
        sig_freq = drive_freq + mod_freq
        sig_freq2 = drive_freq - mod_freq

        print "Signal freq is: ", sig_freq, " Hz"

        cdat, attribs, _ = bu.getdata( f )

        ctime = bu.labview_time_to_datetime(attribs['Time'])

        Fs = attribs['Fsamp']

        cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 

        if( len(tot_psd) == 0 ):
            tot_psd = cpsd
        else:
            tot_psd += cpsd

        npsd += 1

        freq_idx = np.argmin( np.abs( freqs - sig_freq ) )
        freq_idx2 = np.argmin( np.abs( freqs - sig_freq2 ) )
        freq_idx3 = np.argmin( np.abs( freqs - drive_freq ) )

        bw = np.sqrt(freqs[1]-freqs[0]) ## bandwidth

        if(False):
            plt.figure()
            plt.semilogy( freqs, cpsd )
            plt.semilogy( freqs[freq_idx], cpsd[freq_idx], 'rx' )
            plt.semilogy( freqs[freq_idx2], cpsd[freq_idx2], 'rx' )
            plt.xlim([drive_freq - lower_plot_width, drive_freq + upper_plot_width])
            plt.title(str(cpos))
            plt.show()

        dtype = 0 if not "lowfb" in f else 1
        pp2 = np.sqrt(cpsd[freq_idx2])*bw
        pp3 = np.sqrt(cpsd[freq_idx3])*bw
        tot_dat.append( [cpos, np.sqrt(cpsd[freq_idx])*bw, dtype, pp2, pp3, ctime] )

    tot_dat = np.array(tot_dat)
    np.save("plots/chameleon_data_by_run.npy", tot_dat)
    tot_psd_dat = np.vstack([freqs,np.ndarray.flatten(tot_psd)])
    np.save("plots/chameleon_psd.npy", tot_psd_dat)
else:
    
    tot_dat = np.load("plots/chameleon_data_by_run.npy")
    tot_psd_dat = np.load("plots/chameleon_psd.npy")
    npsd = len(tot_dat)
    freqs = tot_psd_dat[0,:]
    tot_psd = tot_psd_dat[1,:]

print "Total traces used: ", npsd


## minimum resolvable force scales down like
## psd [N/rtHz] * 1/sqrt(n) integrations
cpsd = np.sqrt(tot_psd/npsd)

#bw_fac = 1./np.sqrt( 50. * npsd )
bw_fac = bw

plt.figure()
#plt.semilogy( freqs, cpsd*conv_fac*bw_fac )
#plt.semilogy( freqs[freq_idx], cpsd[freq_idx]*conv_fac*bw_fac, 'rx', label="sidebands" )
#plt.semilogy( freqs[freq_idx2], cpsd[freq_idx2]*conv_fac*bw_fac, 'rx' )
#plt.semilogy( freqs[freq_idx3], cpsd[freq_idx3]*conv_fac*bw_fac, 'o', mfc='none', mec='g', label="drive" )
#plt.xlim([drive_freq - lower_plot_width, drive_freq + upper_plot_width])

plt.loglog( freqs, cpsd*conv_fac*bw_fac )
plt.loglog( freqs[freq_idx], cpsd[freq_idx]*conv_fac*bw_fac, 'rx', label="sidebands" )
plt.loglog( freqs[freq_idx2], cpsd[freq_idx2]*conv_fac*bw_fac, 'rx' )
plt.loglog( freqs[freq_idx3], cpsd[freq_idx3]*conv_fac*bw_fac, 'o', mfc='none', mec='g', label="drive" )

#plt.title(str(cpos))

plt.legend(numpoints=1)

#plt.ylim([1e-19, 1e-17])

plt.ylabel("Force PSD [N/rtHz]")    
plt.savefig("plots/drive_spec.pdf")


#tot_dat[:,0] = (80. - tot_dat[:,0]/10000.*80)

# old_dat = np.load("old_data.npy")

# print old_dat
# gpts = np.logical_and(old_dat[0] > 5, old_dat[0] % 2 == 0)
def make_time_plot(time_dat, force_dat, col, tit):

    plt.subplot(2,1,1)
    cfmt =  mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(cfmt)
    plt.plot(time_dat, force_dat*conv_fac, col+"s")
    plt.title(tit)
    plt.ylabel("Force [N]")

    plt.subplot(2,1,2)
    hh, be = np.histogram(force_dat*conv_fac, bins=20)
    plt.step(be[:-1],hh,col,where="post",linewidth=1.5)
    plt.xlabel("Force [N]")
    plt.ylabel("Counts")


fig = plt.figure()
make_time_plot( tot_dat[:,5], tot_dat[:,1], 'k', "Upper sideband" )
plt.savefig("plots/upper_sideband_vs_time.pdf")

fig2 = plt.figure()
make_time_plot( tot_dat[:,5], tot_dat[:,3], 'b', "Lower sideband" )
plt.savefig("plots/lower_sideband_vs_time.pdf")

fig3 = plt.figure()
## take out modulation factor
make_time_plot( tot_dat[:,5], tot_dat[:,4]*0.2, 'r', "Drive" )
plt.savefig("plots/drive_vs_time.pdf")

# fig = plt.figure()
# gpts = tot_dat[:,2] == 0
# plt.plot(range(len(tot_dat[gpts,1])), tot_dat[gpts,1], 'ks-', label="upper sideband")
# plt.plot(range(len(tot_dat[gpts,1])), tot_dat[gpts,3], 'rs-', label="lower sideband")
# plt.plot(range(len(tot_dat[gpts,1])), tot_dat[gpts,4], 'bs-', label="drive")
#gpts = tot_dat[:,2] == 1
#plt.semilogy(tot_dat[gpts,0], tot_dat[gpts,1], 'rs', label="Low FB")

# plt.errorbar(old_dat[0][gpts], old_dat[1][gpts], yerr=0.2*old_dat[1][gpts], fmt='ro', label="Old shielding")

#for i in range( len( tot_dat[:,0] ) ):
#    plt.gca().arrow( tot_dat[i,0], tot_dat[i,1], 0, -tot_dat[i,1]*0.5, fc='k', ec='k', head_width=1, head_length=0.1*tot_dat[i,1])

plt.legend(loc="upper right", numpoints=1)

#plt.xlim([0,50])
#plt.xlabel("Distance from cantilever [$\mu$m]")
plt.ylabel("Force [N]")

#fig.set_size_inches(6,4.5)

#plt.savefig("force_vs_dist.pdf")

plt.show()


