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
from mpl_toolkits.mplot3d import Axes3D

###################################################################################
do_mean_subtract = True  ## subtract mean position from data
do_poly_fit = True  ## fit data to 1/r^2 (for DC bias data)
do_2d_fit = False ## fit data vs position and voltage to 2d function
sep_forward_backward = False ## handle forward and backward separately
match_overlap_region = True ## for multiple overlapping files, match together
#idx_to_plot=[404,406,408,410,412,413,414,415,416]
idx_to_plot = [234,]

diff_dir = None ##'Y' ## if set, take difference between two positions

sig_dir = 'y' ## Direction of the expected signal
pos_offset = 60. ## um, distance of closest approach (needed to make voltage template)

step_channel = 'X'
# sweep channel from drive column in directory file

do_x_sweep = True 
do_y_sweep = False #True
do_z_sweep = False

data_columns = [0,1,2] ## data to plot, x=0, y=1, z=2
mon_columns = [3,7]  # columns used to monitor position, empty to ignore
plot_title = 'Force vs. position'
nbins = 20  ## number of bins vs. bead position

max_files = 50 ## max files to process per directory
force_remake_file = True #False ## force recalculation over all files

buffer_points = 1000 ## number of points to cut from beginning and end of file

make_opt_filt_plot = True
plot_psds = False
dirs_to_plot=['x','y','z']

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/charles/opt_lev/scripts/dir_file.txt" )
###################################################################################

cant_step_per_V = 8. ##um

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
    bin_edges = np.linspace(binmin, binmax, n+1)
    inds = np.digitize(xvec, bin_edges, right = False)-1
    bins = bin_edges[:-1] + np.diff(bin_edges)/2.0
    avs = np.zeros(n)
    ers = np.zeros(n)
    for i in range(len(bins)):
        cidx = inds == i
        if( np.sum(cidx) > 0 ):
            avs[i] = np.median(yvec[cidx])
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

    if( d == 'Z' ):
        coord = re.findall("nmZ\d+nm", s)
        if( len(coord) == 0):
            return None
        else:
            return int(coord[0][3:-2])

def get_pos_from_mon(cmonz, pos_at_10V ):
    return pos_at_10V + cant_step_per_V*(10. - cmonz)

def process_files(data_dir, num_files, step_pos=-999999, pos_at_10V=0., 
                  monmin=0., monmax=10., conv_fac =1., drive_indx=19):

    out_dict = {}

    if( step_pos > -999999 ):
        print "Data with step pos (V): ", step_pos
        if(do_x_sweep):
            flist = sorted(glob.glob(os.path.join(data_dir, "*stageX%dnm*.h5"%abs(step_pos))), key = sort_fun)
        if(do_y_sweep):
            flist = sorted(glob.glob(os.path.join(data_dir, "*nmY%dnm*.h5"%abs(step_pos))), key = sort_fun)
        if(do_z_sweep):
            flist = sorted(glob.glob(os.path.join(data_dir, "*nmZ%dnm*.h5"%abs(step_pos))), key = sort_fun)
        flist1 = []
    else:
        flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        flist1 = []

    tempdata, tempattribs, temphandle = bu.getdata(flist[0])
    drive_freq = tempattribs['stage_settings'][5]
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
            bm,am = signal.butter(3,(drive_freq-4.)/(Fs/2.), btype='highpass')
            #truncdata_x = truncdata_x - np.mean(truncdata_x)
            #truncdata_y = truncdata_y - np.mean(truncdata_y)
            #truncdata_z = truncdata_z - np.mean(truncdata_z)
            truncdata_x = signal.filtfilt(bm,am,cdat[:,data_columns[0]])[buffer_points:-buffer_points]
            truncdata_y = signal.filtfilt(bm,am,cdat[:,data_columns[1]])[buffer_points:-buffer_points]
            truncdata_z = signal.filtfilt(bm,am,cdat[:,data_columns[2]])[buffer_points:-buffer_points]
            
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
        cpos = get_pos_from_mon(cmonz, pos_at_10V)
        cdrive = bu.get_chameleon_force( cpos*1e-6 )
        cdrive -= np.mean(cdrive)
        ## convert newtons to V
        cdrive /= conv_fac

        st = np.fft.rfft( cdrive.flatten() )
        J = np.ones_like( st )
        norm_fac = np.real(np.sum(np.abs(st)**2/J))

        physmin = get_pos_from_mon(monmax, pos_at_10V)
        physmax = get_pos_from_mon(monmin, pos_at_10V)

        ## now bin the data, separating into forward and backward
        for col in ['x','y','z']:
            for mask, sdir in zip([posmask,negmask,allmask],['pos','neg','both']):

                btrace, cerr, bins = bin( cpos[mask], truncdata_dict[col][mask]*conv_fac, 
                                          binmin=physmin, binmax=physmax, n=nbins)
                cname = 'binned_dat_' + col + '_' + sdir

                if( sdir == 'both' ):
                    cpsd,cfreq = mlab.psd(  truncdata_dict[col], 
                                            NFFT=len(truncdata_dict[col]), Fs=Fs )
                    cpsd *= conv_fac**2

                    vt = np.fft.rfft( truncdata_dict[col][mask] )
                    of_amp = np.real( np.sum( np.conj(st)*vt/J ) / norm_fac )

                else:
                    cpsd = []
                    of_amp = 0.

                if( not cname in out_dict ):
                    out_dict[ cname ] = [[btrace,], [cerr,], bins, [of_amp,],[cpsd,]]
                else:
                    out_dict[ cname ][0].append(btrace)
                    out_dict[ cname ][1].append(cerr)
                    out_dict[ cname ][3].append(of_amp)
                    out_dict[ cname ][4].append(cpsd)

    ## we've now looped through all the files, so average everything down
    for col in ['x','y','z']:
        for sdir in ['pos','neg','both']:
            cname = 'binned_dat_' + col + '_' + sdir

            bavg = np.median( np.array(out_dict[cname][0]), axis=0)
            berr = np.sqrt( np.sum(np.array(out_dict[cname][1])**2,axis=0)/len(out_dict[cname][1]) )
            
            tot_psd = np.sqrt( np.sum( out_dict[cname][4],axis=0)/len( out_dict[ cname ][4] ) )

            out_dict[cname + "_avg"] = [out_dict[cname][2], bavg, berr, tot_psd]
    
            ## zero out individual PSDs at this point to minimize file size
            if( sdir == 'both' ):
                out_dict[ cname ][4] = []

    out_dict['freq_list'] = cfreq
                
    return out_dict



## don't correct for the gain, when getting the file name
def get_step_pos_orig(s):  
    if(do_x_sweep):
        stepX = re.findall("stageX\d+",s)
        curr_str = int(stepX[0][6:])
    if(do_y_sweep):
        stepY = re.findall("nmY\d+",s)
        curr_str = int(stepY[0][3:])
    if(do_z_sweep):
        stepZ = re.findall("nmZ\d+",s)
        curr_str = int(stepZ[0][3:])
    return curr_str

def get_step_pos(s):
    if(do_x_sweep):
        stepX = re.findall("stageX\d+",s)
        curr_str = int(stepX[0][6:])
    if(do_y_sweep):
        stepY = re.findall("nmY\d+",s)
        curr_str = int(stepY[0][3:])
    if(do_z_sweep):
        stepZ = re.findall("nmZ\d+",s)
        curr_str = int(stepZ[0][3:])
    return curr_str

data = []
# dir, label, drive_column, numharmonics, monmin, monmax
#  process_files(data_dir, num_files, numharmonics, monmin, monmax,
#                   drive_indx=19):
last_pos_at_10V = 0

for cdir in dirs:

    data_file_path = cdir[0].replace("/data/","/home/charles/analysis/")
    ## make directory if it doesn't exist
    if(not os.path.isdir(data_file_path) ):
        os.makedirs(data_file_path)
    proc_file = os.path.join( data_file_path, "cant_force_vs_position.pkl" )
    file_exists = os.path.isfile( proc_file ) and not force_remake_file


    ## first get a list of all the dc offsets in the directory
    #print cdir
    clist = glob.glob( os.path.join( cdir[0], "*.h5") )
    print cdir[0]
    step_list = []
    step_list_orig = []
    for cf in clist:
        step = get_step_pos( cf )
        step_orig = get_step_pos_orig( cf )
        step_list.append( step  )
        step_list_orig.append( step_orig  )
    step_list, idx = np.unique(step_list, return_index=True)
    step_list_orig = np.array(step_list_orig)[idx]
    print "List of step positions: ", step_list


    if(not file_exists):

        curr_data = []
        for step_pos,step_pos_orig in zip(step_list,step_list_orig):
            print step_pos

            curr_dict = process_files(cdir[0],max_files,drive_indx=cdir[2],step_pos=step_pos_orig,
                                      pos_at_10V=cdir[6],conv_fac=cdir[7])


            #clab = str(80. - cant_step_per_V * step_pos / 1000. + 10) + ' um'
            clab = str(step_pos / 1000.) + ' um'

            curr_dict['label'] = clab
            curr_dict['step_pos'] = step_pos

            curr_data.append( curr_dict )


        out_file = open( proc_file, 'wb')
        pickle.dump(curr_data, out_file)
        out_file.close()
    else:
        print "Loading previously processed data from: %s" % proc_file
        out_file = open( proc_file, 'rb')
        curr_data = pickle.load( out_file )
        out_file.close()

    data += curr_data



###################

## make total list of all dc offsets
def get_dcvolt_from_label(label):
    try:
        dc_volt = float(label[:-5])/1000.
    except ValueError:
        dc_volt = 0.
    return dc_volt

###################



tot_step_list = []
for d in data:
    tot_step_list.append( d['step_pos'] )
tot_step_list = np.unique(tot_step_list)

## make color list for given number of files
colors_yeay = bu.get_color_map( len(data) )
colors_step = bu.get_color_map( len(tot_step_list) )

## power spectra
if( plot_psds ):
    plt.figure(1)
    xlims = [1,100]
    for i,d in enumerate(data):
        label = d['label']
        for j,v in enumerate(dirs_to_plot):
            plt.subplot(len(dirs_to_plot),1,j+1)
            cxdat = d['freq_list']
            cydat = d['binned_dat_'+v+'_both_avg'][3]

            plt.loglog(cxdat, cydat, label=label, color=colors_yeay[i])
            plt.ylabel(v+" PSD [N/rtHz]")   
            plt.xlim(xlims)

    plt.xlabel("Freq [Hz]")   
    plt.legend(loc=0)


def ffn(x, a, b):
    return a * (1./x) + b

def ffn2(x, a, b, c):
    return a * (x - b)**2 + c
 
mag_list = []
data_vs_volt = []
ax_list2 = []
tot_vdat, tot_bdat, tot_fdat = [],[],[]
sub_dat = []
ofig = plt.figure(111)
pfig = plt.figure(112)

steps = []
amplitudes = []

old_offsets = {}
for i,d in enumerate(data):
    label = d['label']
    curr_dat = []
    for j,v in enumerate(dirs_to_plot):
        plt.figure(ofig.number)
        plt.subplot(len(dirs_to_plot),1,j+1)
        if( i == 0 ):
            ax_list2.append( [plt.gca(),] )
        if( sep_forward_backward ):
            cd = d['binned_dat_'+v+'_pos_avg']
            bins, dat, err = cd[0], cd[1], cd[2]
            gpts = dat != 0
            plt.errorbar(bins[gpts], dat[gpts], err[gpts], fmt='o-',label=label, color=colors_yeay[i])

            cd = d['binned_dat_'+v+'_neg_avg']
            bins, dat, err = cd[0], cd[1], cd[2]
            gpts = dat != 0
            plt.errorbar(bins[gpts], dat[gpts], err[gpts], fmt='s-',label=label, color=colors_yeay[i])
            if( do_poly_fit ):
                print "Poly fit requires not to separate forward and back, skipping"
        else:
            ## if there are previous files in this region, make
            ## sure to match the mean in the overlap region
            cd = d['binned_dat_'+v+'_both_avg']
            bins, dat, err = cd[0], cd[1], cd[2]

            is_first_pos = True
            offset = 0.
            curr_step = d['step_pos']

            gpts = dat != 0
            if( not match_overlap_region ):
                offset = -np.mean( dat[gpts] )
            if( len(tot_step_list) > 1 ): 
                coll = colors_step[ np.argwhere( tot_step_list == curr_step)[0] ]
            else:
                coll = colors_yeay[i]
            #offset = -dat[gpts][-1]   # Kludgy way to match data at endpoints
            plt.errorbar(bins[gpts], dat[gpts]+offset, err[gpts], fmt='o-',label=label, color=coll)
            plt.ylabel('Force [N]')
            curr_dat.append(dat[gpts])


            if( do_poly_fit and v == 'y'):
                
                plt.figure(pfig.number)
                plt.errorbar(bins[gpts], dat[gpts]+offset, err[gpts], fmt='o-',label=label, color=coll)
                dc_volt = get_dcvolt_from_label(label)
                xdat, ydat = bins[gpts], dat[gpts]
                tot_vdat.append( dc_volt*np.ones_like(xdat) )
                tot_bdat.append( xdat )
                A, Aerr = opt.curve_fit( ffn, xdat, ydat, p0=[1.,0] )
                mag_list.append([dc_volt,A[0],np.sqrt(Aerr[0,0])])
                tot_fdat.append( ydat - ffn(np.array(bins[gpts])[-1],*A) )
                
                xx = np.linspace( np.min(xdat), np.max(xdat), 1e3 )
                plt.plot( xx, ffn(xx,A[0],A[1]), color=coll, linewidth=1.5)
                fval = ffn(20.,A[0],0)
                print "Fit to %.2fV: A[0]=%e, A[1]=%e, Force[20 um]=%e"%(dc_volt, A[0], A[1],fval)

                if '41.25' in label:
                    continue
                steps.append(float(label[:-3]))
                amplitudes.append(fval)

    sub_dat.append(curr_dat)

plt.xlabel('Cantilever X-position (lateral) [um]')
plt.legend(numpoints=1,loc=0,prop={'size':8})



A, Aerr = opt.curve_fit( ffn2, steps, amplitudes )


plt.figure(69)
plt.plot(steps, amplitudes)

xx = np.linspace(steps[0], steps[-1], 100)
fxx = ffn2(xx, A[0], A[1], A[2])
plt.plot(xx, fxx)

print "Best Fit X-position: ", A[1]


plt.show()
