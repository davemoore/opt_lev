## measure the force from the cantilever, averaging over files
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.interpolate as interp
import scipy.optimize as opt

############################################################
do_mean_subtract = True  ## subtract mean position from data
do_poly_fit = False  ## fit data to 1/r^2 (for DC bias data)
idx_to_plot = [23,24,25] ## indices from dir file below to use

plot_title = 'Force vs. position'
nbins = 80  ## number of bins vs. bead position

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/dcmoore/opt_lev/scripts/cant_force/dir_file.txt" )
############################################################

dirs = []
# dir, label, drive_column, numharmonics, monmin, monmax, closest_app, cal_fac
for idx in idx_to_plot:
    dirs.append( ddict[str(idx)] )
print dirs

sbins = 4  # number of bins to either side of drive_freq to integrate

def sort_fun( s ):
    return float(re.findall("\d+.h5", s)[0][:-3])


def bin(xvec, yvec, binmin=0, binmax=10, n=300):
    bins = np.linspace(binmin, binmax, n)
    inds = np.digitize(xvec, bins, right = False)
    avs = np.zeros(n)
    ers = np.zeros(n)
    for i in range(len(bins)):
        cidx = inds == i
        if( np.sum(cidx) > 0 ):
            avs[i] = np.mean(yvec[cidx])
            ers[i] = np.std(yvec[cidx])/np.sqrt(len(yvec[cidx]))

    return avs, ers, bins 


def process_files(data_dir, num_files, numharmonics, \
                  monmin, monmax, drive_indx=19, conv_fac=1., dc_val=-1):
    ## Load a series of files, acausal filter the cantilever drive and 
    ## some number of harmonics then bin the data and plot position/force
    ## as a function of cantilever position
    global sbins
    global nbins

    if( dc_val >= 0 ):
        print dc_val
        flist = sorted(glob.glob(os.path.join(data_dir, "*%dmVdc*.h5"%dc_val)), key = sort_fun)
    else:
        flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)

    tempdata, tempattribs, temphandle = bu.getdata(flist[0])

    drive_freq = tempattribs['stage_settings'][-2]

    temphandle.close()     

    binned_tracesf = []
    binned_errorsf = []
    binned_tracesr = []
    binned_errorsr = []
    tot_psdi = []
    tot_psdf = []

    ntrace = 0

    for fidx,f in enumerate(flist[:num_files]):

        print f 
        ## Load the data
        cdat, attribs, _ = bu.getdata( f )
        cmonz = cdat[:,drive_indx][1000:len(cdat[:,drive_indx])-1000] 
        # if( "nomon" in f ):
        #     cdat2, attribs2, _ = bu.getdata( "/data/20151021/nobead_withaperture_farout/1_5mbar_nobead_stageX0nmY6000nmZ4000nmZ4000mVAC4Hz_0.h5" )            
        #     cmonz = cdat2[:,drive_indx][1000:len(cdat2[:,drive_indx])-1000] 
        truncdata = cdat[:,1][1000:len(cdat[:,1])-1000]
        Fs = attribs['Fsamp']        
        ntrace += 1

        ## Compute the FFT and frequency bins
        cfft = np.fft.rfft(truncdata)
        freqs = np.fft.rfftfreq(len(truncdata), d=1.0/Fs) 

        ## Find indices in FFT at drive freq and harmonics
        indices = []
        for i in range(numharmonics+1)[1:]:
            indx = np.argmin(np.abs( freqs - i * drive_freq))
            indices.append(indx)

        ## Construct the acausal filtered FFT array and include DC component
        fft_filt = 0 * np.copy(cfft)
        #fft_filt[0] = cfft[0]

        ## Apply the acausal filter
        for indx in indices:
            fft_filt[indx-sbins:indx+sbins+1] = cfft[indx-sbins:indx+sbins+1]

        ## IFFT to obtain filtered signal and add to tot signal and tot_mon
        ctrace = np.fft.irfft(fft_filt)

        ## Consider stage travel direction separately
        ## filter the monitor around the drive freq
        b,a = signal.butter(3,(drive_freq+2.)/(Fs/2.), btype='lowpass')
        cmonz_filt = signal.filtfilt( b, a, cmonz )
        monderiv = np.gradient(cmonz_filt)
        posmask = monderiv >= 0
        negmask = monderiv < 0 

        # plt.figure()
        # plt.plot(cmonz)
        # plt.plot(cmonz_filt)
        # plt.plot(truncdata)
        # plt.plot(monderiv)
        # plt.show()

        ## Subtract the mean to compensate for long time drift
        if(do_mean_subtract):
            truncdata = truncdata - np.mean(truncdata)

        #btrace, cerr, bins = bin(cmonz, ctrace, \
        #                         binmin=monmin, binmax=monmax, n=300)
        btracef, cerrf, binsf = bin(cmonz[posmask], truncdata[posmask], \
                                    binmin=monmin, binmax=monmax, n=nbins)
        btracer, cerrr, binsr = bin(cmonz[negmask], truncdata[negmask], \
                                    binmin=monmin, binmax=monmax, n=nbins)
        bmon, monerr, monbins = bin(cmonz, monderiv, binmin=monmin, \
                                    binmax=monmax, n=nbins)



        binned_tracesf.append(btracef)
        binned_tracesr.append(btracer)

        binned_errorsf.append(cerrf)
        binned_errorsr.append(cerrr)

        ## Add to the PSDs
        if( len(tot_psdi) == 0 ):
            tot_psdi = cfft * cfft.conj()
            tot_psdf = fft_filt * fft_filt.conj()
        else:
            tot_psdi += cfft * cfft.conj()
            tot_psdf += fft_filt * fft_filt.conj()

    binned_tracesf = np.array(binned_tracesf)
    binned_errorsf = np.array(binned_errorsf)
    binned_tracesr = np.array(binned_tracesr)
    binned_errorsr = np.array(binned_errorsr)

    avsf = np.mean(binned_tracesf, axis=0)
    ersf = np.sqrt(np.sum(binned_errorsf**2, axis=0) \
                   / np.shape(binned_errorsf)[0])
    avsr = np.mean(binned_tracesr, axis=0)
    ersr = np.sqrt(np.sum(binned_errorsr**2, axis=0) \
                   / np.shape(binned_errorsr)[0])
        
    tot_psdi = tot_psdi * (1. / ntrace)
    tot_psdf = tot_psdf * (1. / ntrace)

    return binsf, binsr, avsf, avsr, ersf, ersr, freqs, tot_psdi, tot_psdf

def get_dc_offset(s):
    dcstr = re.findall("\d+mVdc", s)
    if( len(dcstr) == 0 ):
        return -1
    else:
        return int( dcstr[0][:-4] )

data = []
# dir, label, drive_column, numharmonics, monmin, monmax
#  process_files(data_dir, num_files, numharmonics, monmin, monmax,
#                   drive_indx=19):
for cdir in dirs:

    ## first get a list of all the dc offsets in the directory
    #print cdir
    clist = glob.glob( os.path.join( cdir[0], "*.h5") )
    dc_list = []
    for cf in clist:
        dc_list.append( get_dc_offset( cf ) )
    dc_list = np.unique(dc_list)

    for dc_val in dc_list:
        print dc_val

        binsf, binsr, avsf, avsr, ersf, ersr, freqs, psdi, psdf = \
                process_files(cdir[0], 1000, cdir[3], cdir[4], cdir[5], cdir[2], dc_val=dc_val)
        volts = cdir[5] - cdir[4]
        ums = volts * 8
        binsf = cdir[6]+8.*(volts - binsf)
        binsr = cdir[6]+8.*(volts - binsr)

        conv_fac = cdir[7]
        avsf *= conv_fac
        avsr *= conv_fac
        ersf *= conv_fac
        ersr *= conv_fac
        if( dc_val >= 0 ):
            clab = str(dc_val) + " mV DC"
        else:
            clab = cdir[1]
        data.append([binsf, binsr, avsf, avsr, ersf, ersr, freqs, psdi, psdf, clab])

plt.figure(1)
for i in range(len(data)):
    #label = dirs[i][1]
    label = data[i][9]
    plt.loglog(data[i][6], np.abs(data[i][7]), label=label)
    plt.loglog(data[i][6], np.abs(data[i][8]))
plt.legend(loc=0)


plt.figure(2)
g = plt.gcf()
plot = plt.subplot(111)
plot.tick_params(axis='both', labelsize=16)

## function to fit data vs position
def ffn(x,A,B):
    return A * (1./x)**2 + B

## function to fit force vs voltage
def ffn2(x,A):
    return A * x**2

colors_yeay = ['b', 'r', 'g', 'k', 'c', 'm', 'y']
mag_list = []
for i in range(len(data)):
    #label = dirs[i][1]
    #print data[i][0], data[i][1], data[i][2]
    label = data[i][9]
    gpts = data[i][4] != 0
    plt.errorbar(data[i][0][gpts], data[i][2][gpts], data[i][4][gpts], fmt='o-', \
                 label=label, color=colors_yeay[i])
    gpts = data[i][5] != 0
    plt.errorbar(data[i][1][gpts], data[i][3][gpts], data[i][5][gpts], fmt='o-', \
                 color=colors_yeay[i])

    ## fit 1/r^3 to the dipole response
    if( do_poly_fit ):
        A, Aerr = opt.curve_fit( ffn, data[i][0][gpts], data[i][2][gpts], p0=[1.,0] )
        dc_volt = float(label[:-5])/1000.
        mag_list.append([dc_volt,A[0],np.sqrt(Aerr[0,0])])
        xx = np.linspace( np.min(data[i][0][gpts]), np.max(data[i][0][gpts]), 1e3 )
        plt.plot( xx, ffn(xx,A[0],A[1]), color=colors_yeay[i], linewidth=1.5)

plt.xlabel('Distance From Bead [um]', fontsize='16')
if( do_mean_subtract ):
    plt.ylabel('Force [N]', fontsize='16')
else:
    plt.ylabel('Mean Subtracted Force [N]', fontsize='16')
plt.title(plot_title, fontweight='bold', fontsize='16', y=1.05)
#plt.xlim(30,110)
plt.legend(loc=0)

g.set_size_inches(8,6)
#plt.savefig('force-v-date.pdf')
#plt.ylim(-1.4e-15, 1e-15)
#plt.savefig('force-v-pos_multipressure2.pdf')

if( do_poly_fit ):
    mag_list = np.array(mag_list)
    fit_fig = plt.figure()
    plt.errorbar( mag_list[:,0], mag_list[:,1], yerr=mag_list[:,2], fmt='ko' )
    A, Aerr = opt.curve_fit( ffn2, mag_list[:,0], mag_list[:,1], p0=[1.,] )    
    xx = np.linspace( np.min(mag_list[:,0]), np.max(mag_list[:,0]), 1e3 )
    plt.plot(xx, ffn2(xx, A[0]), 'r', linewidth=1.5)

    plt.xlabel("Cantilever DC bias [V]")
    plt.ylabel("Force from fit at 1um [N]")


plt.show()
