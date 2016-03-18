## measure the force from the cantilever, averaging over files
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.signal as signal
import scipy.interpolate as interp
import scipy.optimize as opt
import cPickle as pickle

###################################################################################
do_mean_subtract = True  ## subtract mean position from data
do_poly_fit = False  ## fit data to 1/r^2 (for DC bias data)
do_2d_fit = True ## fit data vs position and voltage to 2d function
sep_forward_backward = False ## handle forward and backward separately
idx_to_plot = [217,218] #[217,218,219,221,222] ## indices from dir file below to use
diff_dir = None ##'Y' ## if set, take difference between two positions

data_columns = [0,1,2] ## data to plot, x=0, y=1, z=2
mon_columns = [3,7]  # columns used to monitor position, empty to ignore
plot_title = 'Force vs. position'
nbins = 40  ## number of bins vs. bead position

max_files = 10 ## max files to process per directory
force_remake_file = True ## force recalculation over all files

buffer_points = 1000 ## number of points to cut from beginning and end of file

make_opt_filt_plot = True

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/arider/opt_lev/scripts/cant_force/dir_file.txt" )
###################################################################################

cant_step_per_V = 8. ##um
colors_yeay = ['b', 'r', 'g', 'k', 'c', 'm', 'y', [0.5,0.5,0.5], [0, 0.5, 1], [1, 0.5, 0], [0.5, 1,0]]

dirs = []
# dir, label, drive_column, numharmonics, monmin, monmax, closest_app, cal_fac
for idx in idx_to_plot:
    dirs.append( ddict[str(idx)] )
print dirs

sbins = 4  # number of bins to either side of drive_freq to integrate

def sort_fun( s ):
    cc = re.findall("-?\d+.h5", s)
    if( len(cc) > 0 ):
        return float(cc[0][:-3])
    else:
        return -1.

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

def get_stage_dir_pos( s, d ):
    if( d == 'X' ):
        coord = re.findall("stageX\d+nm", s)
        if( len(coord) == 0 ):
            return None
        else:
            return int(coord[0][6:-2])

    if( d == 'Y' ):
        coord = re.findall("nmY\d+nm", s)
        if( len(coord) == 0 ):
            return None
        else:
            return int(coord[0][3:-2])

def process_files(data_dir, num_files, dc_val=-999999, pos_at_10V=0., 
                  monmin=20., monmax=100., conv_fac =1., drive_indx=19):

    out_dict = {}

    if( dc_val > -999999 ):
        print "Data with DC bias (V): ", dc_val
        flist = sorted(glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%dc_val)), key = sort_fun)
        if( len( flist ) == 0 ):
            ## must be the dc supply
            flist = sorted(glob.glob(os.path.join(data_dir, "*dcps%dmVdc*.h5"%dc_val)), key = sort_fun)
        flist1 = []
    else:
        flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        flist1 = []

    tempdata, tempattribs, temphandle = bu.getdata(flist[0])
    drive_freq = tempattribs['stage_settings'][-2]
    temphandle.close()  


    for fidx,f in enumerate(flist[:num_files]):

        print f 
        ## Load the data
        cdat, attribs, fhand = bu.getdata( f )
        if( len(cdat) == 0):
            print "Skipping: ", f
            continue         

        Fs = attribs['Fsamp']        

        ## get the data and cut off the beginning and the end to avoid edge effects
        cmonz = cdat[:,drive_indx][buffer_points:-buffer_points] 
        truncdata_x = cdat[:,data_columns[0]][buffer_points:-buffer_points]
        truncdata_y = cdat[:,data_columns[1]][buffer_points:-buffer_points]
        truncdata_z = cdat[:,data_columns[2]][buffer_points:-buffer_points]

        ## Subtract the mean to compensate for long time drift
        if(do_mean_subtract):
            truncdata_x = truncdata_x - np.mean(truncdata_x)
            truncdata_y = truncdata_y - np.mean(truncdata_y)
            truncdata_z = truncdata_z - np.mean(truncdata_z)

        truncdata_dict = {'x': truncdata_x, 
                          'y': truncdata_y, 
                          'z': truncdata_z}

        ## Consider stage travel direction separately
        ## filter the monitor around the drive freq
        b,a = signal.butter(3,(drive_freq+2.)/(Fs/2.), btype='lowpass')
        cmonz_filt = signal.filtfilt( b, a, cmonz )
        monderiv = np.gradient(cmonz_filt)
        posmask = monderiv >= 0
        negmask = monderiv < 0 
        allmask = np.logical_or(posmask, negmask)

        ## optimal filter 
        cpos = pos_at_10V + cant_step_per_V*(10. - cmonz)
        cdrive = bu.get_chameleon_force( cpos*1e-6 )
        cdrive -= np.mean(cdrive)
        ## convert newtons to V
        cdrive /= conv_fac

        st = np.fft.rfft( cdrive.flatten() )
        J = np.ones_like( st )
        norm_fac = np.real(np.sum(np.abs(st)**2/J))

        ## now bin the data, separating into forward and backward
        for col in ['x','y','z']:
            for mask, sdir in zip([posmask,negmask,allmask],['pos','neg','both']):

                btrace, cerr, bins = bin( cpos[mask], truncdata_dict[col][mask]*conv_fac, 
                                          binmin=monmin, binmax=monmax, n=nbins)
                cname = 'binned_dat_' + col + '_' + sdir

                if( sdir == 'both' ):
                    cpsd,cfreq = mlab.psd(  truncdata_dict[col], 
                                            NFFT=len(truncdata_dict[col]), Fs=Fs )
                    cpsd *= conv_fac**2

                    vt = np.fft.rfft( truncdata_dict[col][mask] )
                    of_amp = np.real( np.sum( np.conj(st)*vt/J ) / norm_fac )

                else:
                    cpsd, cfreq = [],[]
                    of_amp = 0.

                if( not cname in out_dict ):
                    out_dict[ cname ] = [[btrace,], [cerr,], bins, [of_amp,],[cpsd,],cfreq]
                else:
                    out_dict[ cname ][0].append(btrace)
                    out_dict[ cname ][1].append(cerr)
                    out_dict[ cname ][3].append(of_amp)
                    out_dict[ cname ][4].append(cpsd)

    ## we've now looped through all the files, so average everything down
    for col in ['x','y','z']:
        for sdir in ['pos','neg','both']:
            cname = 'binned_dat_' + col + '_' + sdir

            bavg = np.mean( np.array(out_dict[cname][0]),axis=0)
            berr = np.sqrt( np.sum(np.array(out_dict[cname][1])**2,axis=0)/len(out_dict[cname][1]) )
            
            tot_psd = np.sqrt( np.sum( out_dict[ cname ][4],axis=0)/len( out_dict[ cname ][4] ) )

            out_dict[cname + "_avg"] = [out_dict[cname][2], bavg, berr, tot_psd, cfreq]
    
    return out_dict


def process_files_old(data_dir, num_files, numharmonics, \
                  monmin, monmax, drive_indx=19, dc_val=-1, pos_at_10V=0., conv_fac=1.):
    ## Load a series of files, acausal filter the cantilever drive and 
    ## some number of harmonics then bin the data and plot position/force
    ## as a function of cantilever position
    global sbins
    global nbins
    
    if( diff_dir ):
        ## figure out what values of the dir position exist
        init_list = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        dir_coords = np.unique([get_stage_dir_pos(f,diff_dir) for f in init_list])
        
        flist = sorted(glob.glob(os.path.join(data_dir, "*%s%dnm*.h5"%(diff_dir,dir_coords[0]))), key = sort_fun)
        flist1 = sorted(glob.glob(os.path.join(data_dir, "*%s%dnm*.h5"%(diff_dir,dir_coords[1]))), key = sort_fun)
        ## make sure we have exactly the same number of files
        flist = flist[:len(flist1)]

    elif( dc_val > -999999 ):
        print dc_val
        flist = sorted(glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%dc_val)), key = sort_fun)
        if( len( flist ) == 0 ):
            ## must be the dc supply
            flist = sorted(glob.glob(os.path.join(data_dir, "*dcps%dmVdc*.h5"%dc_val)), key = sort_fun)
        flist1 = []
    else:
        flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        flist1 = []

    tempdata, tempattribs, temphandle = bu.getdata(flist[0])

    drive_freq = tempattribs['stage_settings'][-2]

    temphandle.close()     

    binned_traces = []
    binned_errors = []
    binned_tracesf = []
    binned_errorsf = []
    binned_tracesr = []
    binned_errorsr = []
    tot_psdi = []
    tot_psdf = []
    of_amp_list = []

    ntrace = 0

    for fidx,f in enumerate(flist[1:num_files]):

        print f 
        ## Load the data
        cdat, attribs, fhand = bu.getdata( f )
        if( len(cdat) == 0):
            print "Skipping: ", f
            continue         

        Fs = attribs['Fsamp']        
        cmonz = cdat[:,drive_indx][buffer_points:-buffer_points] 
        truncdata = cdat[:,data_column][buffer_points:-buffer_points]

        ## check correlation with beam used to monitor trap tilt
        if( len(mon_columns) == 2 and not diff_dir):
            mx, my = get_mon_amp_and_phase( cdat, drive_freq, Fs )
            
            if( False ):
                b2,a2 = signal.butter(1,[(drive_freq-0.1)/(Fs/2),(drive_freq+0.1)/(Fs/2)], btype='bandpass')
                plt.figure()
                plt.plot(signal.filtfilt(b2,a2,truncdata))
                plt.plot(mx)
                #plt.plot(my)
                
                # plt.figure()
                # plt.plot(truncdata)
                # plt.plot(truncdata - (mx + my) )

                plt.show()
            
            #truncdata -= (mx + my)
            #truncdata -= (mx)

            ## chop off windowing artifacts
            #truncdata = truncdata[2*buffer_points:-2*buffer_points]
            #cmonz = cmonz[2*buffer_points:-2*buffer_points]


        ## if we want to take the difference with a reference file, check that the file
        ## exists, and if so, subtract off the reference                             
        if( diff_dir ):
            cdat1, attribs1, fhand1 = bu.getdata( flist1[fidx] )   
            if( len(cdat1) == 0 ):
                print "Couldn't find matching file: ", f
                continue
            truncdata_diff = cdat1[:,data_column][buffer_points:-buffer_points]
            if( len(mon_columns) == 2 ):
                mxd, myd = get_mon_amp_and_phase( cdat1, drive_freq, Fs )

                truncdata -= (mxd + myd)
            else:
                truncdata -= truncdata_diff
            
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

        ## optimal filter 
        cpos = pos_at_10V + cant_step_per_V*(10. - cmonz)
        cdrive = bu.get_chameleon_force( cpos*1e-6 )
        #cdrive = np.ones_like(cpos)
        cdrive -= np.mean(cdrive)
        ## convert newtons to V
        cdrive /= conv_fac


        ## add some fake signal in
        #truncdata += cdrive*0.00001

        #btrace, cerr, bins = bin(cmonz, ctrace, \
        #                         binmin=monmin, binmax=monmax, n=300)
        btracef, cerrf, binsf = bin(cmonz[posmask], truncdata[posmask], \
                                    binmin=monmin, binmax=monmax, n=nbins)
        btracer, cerrr, binsr = bin(cmonz[negmask], truncdata[negmask], \
                                    binmin=monmin, binmax=monmax, n=nbins)
        btrace, cerr, bins = bin(cmonz, truncdata, \
                                    binmin=monmin, binmax=monmax, n=nbins)
        bmon, monerr, monbins = bin(cmonz, monderiv, binmin=monmin, \
                                    binmax=monmax, n=nbins)

        vt = np.fft.rfft( truncdata )
        #vt_diff = np.fft.rfft( truncdata_diff )
        st = np.fft.rfft( cdrive.flatten() )
        J = np.ones_like( vt )

        ## look at max opt filt output
        # xvals = np.arange(-20000,10000,10)
        # efac = -2*np.pi*1j*(xvals)/len(st)
        # of_vec = []
        # for ei in efac:
        #     of_vec.append(np.real( np.sum( np.conj(st)*vt*np.exp(ei)/J ) / np.sum(np.abs(st)**2/J) ))
        # of_vec = np.array(of_vec)
        # plt.figure()
        # plt.plot(xvals, of_vec)
        # plt.show()

        of_amp = np.real( np.sum( np.conj(st)*vt/J ) / np.sum(np.abs(st)**2/J) )
        #of_amp_diff = np.real( np.sum( np.conj(st)*vt_diff/J ) / np.sum(np.abs(st)**2/J) )
        of_amp_list.append(of_amp)

        # plt.figure()
        # plt.plot(  truncdata, '.' )
        # dl, dh = (drive_freq-1.)/(Fs/2.), (drive_freq+1.)/(Fs/2.)
        # b2,a2 = signal.butter(1,[dl,dh], btype='bandpass')
        # tf = signal.filtfilt(b2,a2,truncdata)
        # plt.plot(  tf, 'c.' )
        # plt.plot( cdrive * of_amp, 'r' )
        # plt.show()

        binned_tracesf.append(btracef)
        binned_tracesr.append(btracer)
        binned_traces.append(btrace)

        binned_errorsf.append(cerrf)
        binned_errorsr.append(cerrr)
        binned_errors.append(cerr)

        ## Add to the PSDs
        bw_fac = 1. ##2.0/(len(cfft)*Fs)
        if( len(tot_psdi) == 0 ):
            p1,f1 = mlab.psd(  truncdata, NFFT=len(truncdata), Fs=Fs )
            tot_psdi = p1 * conv_fac**2
            #tot_psdi = bw_fac*cfft * cfft.conj() * conv_fac**2
            tot_psdf = bw_fac*fft_filt * fft_filt.conj() * conv_fac**2
        else:
            p2,f2 = mlab.psd(  truncdata, NFFT=len(truncdata), Fs=Fs )
            tot_psdi += p2*conv_fac**2
            #tot_psdi += bw_fac * cfft * cfft.conj() * conv_fac**2
            tot_psdf += bw_fac * fft_filt * fft_filt.conj() * conv_fac**2

        fhand.close()

    binned_tracesf = np.array(binned_tracesf)
    binned_errorsf = np.array(binned_errorsf)
    binned_tracesr = np.array(binned_tracesr)
    binned_errorsr = np.array(binned_errorsr)
    binned_traces = np.array(binned_traces)
    binned_errors = np.array(binned_errors)
    of_amp_list = np.array(of_amp_list)

    avsf = np.mean(binned_tracesf, axis=0)
    ersf = np.sqrt(np.sum(binned_errorsf**2, axis=0) \
                   / np.shape(binned_errorsf)[0])
    avsr = np.mean(binned_tracesr, axis=0)
    ersr = np.sqrt(np.sum(binned_errorsr**2, axis=0) \
                   / np.shape(binned_errorsr)[0])
    avs = np.mean(binned_traces, axis=0)
    ers = np.sqrt(np.sum(binned_errors**2, axis=0) \
                   / np.shape(binned_errors)[0])

    tot_psdi = tot_psdi * (1. / ntrace)
    tot_psdf = tot_psdf * (1. / ntrace)

    return binsf, binsr, avsf, avsr, ersf, ersr, freqs, tot_psdi, tot_psdf, avs, ers, bins, of_amp_list

def get_dc_offset(s):
    dcstr = re.findall("-?\d+mVdc", s)
    if( len(dcstr) == 0 ):
        return -999999
    else:
        return int( dcstr[0][:-4] )

data = []
# dir, label, drive_column, numharmonics, monmin, monmax
#  process_files(data_dir, num_files, numharmonics, monmin, monmax,
#                   drive_indx=19):
for cdir in dirs:

    data_file_path = cdir[0].replace("/data/","/home/arider/analysis/")
    ## make directory if it doesn't exist
    if(not os.path.isdir(data_file_path) ):
        os.makedirs(data_file_path)
    proc_file = os.path.join( data_file_path, "cant_force_vs_position.pkl" )
    file_exists = os.path.isfile( proc_file ) and not force_remake_file


    ## first get a list of all the dc offsets in the directory
    #print cdir
    clist = glob.glob( os.path.join( cdir[0], "*.h5") )
    dc_list = []
    for cf in clist:
        dcoffset = get_dc_offset( cf )
        dc_list.append( dcoffset  )
    dc_list = np.unique(dc_list)
    print "List of dc offsets: ", dc_list

    if(not file_exists):

        curr_data = []
        for dc_val in dc_list:
            print dc_val

            #binsf, binsr, avsf, avsr, ersf, ersr, freqs, psdi, psdf, avs, ers, bins, of_amp_list = \
            #        process_files(cdir[0], max_files, cdir[3], cdir[4], cdir[5], drive_indx=cdir[2], dc_val=dc_val,pos_at_10V=cdir[6],conv_fac=cdir[7])

            curr_dict = process_files(cdir[0],max_files,drive_indx=cdir[2],dc_val=dc_val,
                                      pos_at_10V=cdir[6],conv_fac=cdir[7])

            if( dc_val > -999999 and True):
                clab = str(dc_val) + " mV DC"
            else:
                clab = cdir[1]
            
            curr_dict['label'] = clab

            curr_data.append( curr_dict )


        out_file = open( proc_file, 'wb')
        pickle.dump(curr_data, out_file)
        out_file.close()
    else:
        print "Loading previously processed data from: %s" % proc_file
        curr_data = pickle.load( out_file )

    data += curr_data


## power spectra
plt.figure(1)
for i,d in enumerate(data):
    plt.subplot(3,1,1)
    label = d['label']
    plt.loglog(d['binned_dat_x_both_avg'][4],d['binned_dat_x_both_avg'][3], label=label, color=colors_yeay[i])
    plt.subplot(3,1,2)
    plt.loglog(d['binned_dat_y_both_avg'][4],d['binned_dat_y_both_avg'][3], color=colors_yeay[i])
    plt.subplot(3,1,3)
    plt.loglog(d['binned_dat_z_both_avg'][4],d['binned_dat_z_both_avg'][3], color=colors_yeay[i])
    plt.xlim([1,100])
    plt.ylabel("Force PSD [N/rtHz]")   
    plt.xlabel("Freq [Hz]")
   
plt.legend(loc=0)

plt.show()

of_fig = plt.figure()
for i in range(len(data)):
    label = data[i]['label']
    of_amps = data[i]['binned_dat_y_both'][3]

    if( len(of_amps) > 2 and make_opt_filt_plot):
        bu.make_histo_vs_time( range(len(of_amps)), of_amps,lab=label,col=colors_yeay[i] )
    else:
        plt.plot(range(len(of_amps)), of_amps, 'o-', label=label, color=colors_yeay[i])
        

    ## make a sideways histogram
plt.ylabel('beta value')
plt.legend(loc=0)

plt.figure(2)
g = plt.gcf()
plot = plt.subplot(111)
plt.ylabel("Beta")
#ax1.set_xlabel('file number')
#ax2.set_ylabel('beta value')
plot.tick_params(axis='both', labelsize=16)
plt.show()
## function to fit data vs position
def ffn(x,A,B):
    #return A * (1./x)**2 + B
    return A * (1./(x+50.))**2 + B
    ##return A * (1./(x+15.))**1 + B

## function to fit force vs voltage
def ffn2(x,A):
    return A * x**2

mag_list = []
data_vs_volt = []
for i in range(len(data)):
    #label = dirs[i][1]
    #print data[i][0], data[i][1], data[i][2]
    label = data[i]['label']
    if( sep_forward_backward ):
                plt.errorbar(data[i]['binned_dat_y_pos_avg'][0], data[i]['binned_dat_y_pos_avg'][1], data[i]['binned_dat_y_pos_avg'][2], fmt='o-', label=label, color=colors_yeay[i])
                plt.errorbar(data[i]['binned_dat_y_neg_avg'][0], data[i]['binned_dat_y_neg_avg'][1], data[i]['binned_dat_y_neg_avg'][2], fmt='o-', label=label, color=colors_yeay[i])
    
    else:
        plt.errorbar(data[i]['binned_dat_y_neg_avg'][0], data[i]['binned_dat_y_neg_avg'][1], data[i]['binned_dat_y_neg_avg'][2], fmt='o-', label=label, color=colors_yeay[i])
    
    #data_vs_volt.append( [float(label[:-5])/1000., data[i][10][gpts], data[i][11][gpts]] )
    ## fit 1/r^2 to the dipole response
    if( do_poly_fit ):
        if(sep_forward_backward):
            xdat, ydat = data[i][0][gpts], data[i][2][gpts]
        else:
            xdat, ydat = data[i][10][gpts], data[i][11][gpts]
        A, Aerr = opt.curve_fit( ffn, xdat, ydat, p0=[1.,0] )

        try:
            dc_volt = float(label[:-5])/1000.
        except ValueError:
            dc_volt = 0.
        
        mag_list.append([dc_volt,A[0],np.sqrt(Aerr[0,0])])
        xx = np.linspace( np.min(xdat), np.max(xdat), 1e3 )
        plt.plot( xx, ffn(xx,A[0],A[1]), color=colors_yeay[i], linewidth=1.5)

        print "Fit to %.2fV: A[0]=%e, A[1]=%e"%(dc_volt, A[0], A[1]) 

plt.xlabel('Distance From Bead [um]', fontsize='16')
if( do_mean_subtract ):
    plt.ylabel('Force [N]', fontsize='16')
else:
    plt.ylabel('Force[N]', fontsize='16')
plt.title(plot_title, fontweight='bold', fontsize='16', y=1.05)
#plt.xlim(30,110)
plt.legend(loc=0, numpoints=1)

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


#if( do_2d_fit ):

plt.show()
