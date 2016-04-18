## load skimmed files from cant_force_vs_pos and do the final fit to the chameleon + bkg model

import glob, os, re, sys
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cPickle as pickle
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

bead_1_dict = {"cham_idx": [371,373,375,377,378,379,380,381,382,383,384,385],
               "cal_idx": [370,372,374,376],
               "dc_volt_to_use": 4.,
               "cant_pos_file": "",
               "bead_label": r"$\mu$Sphere 1"}
bead_2_dict = {"cham_idx": [387,389,391,393,395,396,397,398,399,400,401,402],
               "cal_idx": [388,390,392,394],
               "dc_volt_to_use": 4.,
               "cant_pos_file": "",
               "bead_label": r"$\mu$Sphere 2"}
bead_3_dict = {"cham_idx": [404,406,408,410,412,413,414,415,416,418,419],
               "cal_idx": [405,407,409,411],
               "dc_volt_to_use": 4.,
               "cant_pos_file": "",
               "bead_label": r"$\mu$Sphere 3"}

bead_dicts = [bead_1_dict, bead_2_dict, bead_3_dict]
#bead_dicts = [bead_3_dict,]

do_reposition = True
pos_offsets = [-7.7,10.9,13.5,1.7]

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/dcmoore/opt_lev/scripts/cant_force/dir_file.txt" )

def get_dcvolt_from_label(label):
    try:
        dc_volt = float(label[:-5])/1000.
    except ValueError:
        dc_volt = 0.
    return dc_volt

def get_data(idx_to_plot,is_cal=False):
    dirs = []
    # dir, label, drive_column, numharmonics, monmin, monmax, closest_app, cal_fac
    for idx in idx_to_plot:
        dirs.append( ddict[str(idx)] )

    data = []
    for cdir in dirs:

        data_file_path = cdir[0].replace("/data/","/home/dcmoore/analysis/")
        proc_file = os.path.join( data_file_path, "cant_force_vs_position.pkl" )
        file_exists = os.path.isfile( proc_file )

        if( not file_exists ):
            print "Preprocessed file does not exist for %s" % cdir
            print "Exiting..."
            sys.exit(1)

        print "Loading previously processed data from: %s" % proc_file
        out_file = open( proc_file, 'rb')
        curr_data = pickle.load( out_file )
        out_file.close()

        data += curr_data

    ### Calculate averaged residuals, ignoring stitching ranges together ###
    bin_list = []
    dat_list = []
    err_list = []
    for i,d in enumerate(data):
        label = d['label']
        v = 'y'

        dc_volt = get_dcvolt_from_label(label)
        if(is_cal and dc_volt != dc_volt_to_use/200.): continue

        cd = d['binned_dat_'+v+'_both_avg']
        bins, dat, err = cd[0], cd[1], cd[2]
        gpts = dat != 0

        bins, dat, err = bins[gpts], dat[gpts], err[gpts]

        bin_list.append(bins)
        dat_list.append(dat)
        err_list.append(err)

    bin_list, dat_list, err_list = np.array(bin_list), np.array(dat_list), np.array(err_list)

    ubins = np.unique(bin_list)
    udat = np.zeros_like(ubins)
    uerr = np.zeros_like(ubins)

    for i,b in enumerate(ubins):

        cpos = bin_list == b
        cdat = dat_list[cpos]
        cerr = err_list[cpos]

        udat[i] = np.mean(cdat)
        uerr[i] = np.std(cdat)/np.sqrt( len(cdat) )

    udat -= np.mean(udat)

    min_list = np.unique(bin_list[:,0])[::-1]

    ### Residuals with stitching ################
    bins_avgr = []
    dat_avgr = []
    err_avgr = []
    for j,b in enumerate(min_list):

        curr_rows = bin_list[:,0] == b

        bins_avgr.append( bin_list[np.argwhere(curr_rows)[0][0],:] )
        dat_avgr.append( np.mean(dat_list[curr_rows,:], axis=0) )
        err_avgr.append( np.std(dat_list[curr_rows,:], axis=0)/np.sqrt( np.sum(curr_rows) ) )

    bins_avgr, dat_avgr, err_avgr = np.array(bins_avgr), np.array(dat_avgr), np.array(err_avgr)


    last_offset = 0.
    out_dat = np.zeros( (np.shape(bins_avgr)[0], np.shape(bins_avgr)[1], 3) )
    for i in range( np.shape(bins_avgr)[0] ):
        if( i > 0 ):
            overlap_points_new = bins_avgr[i,:] >= bins_avgr[i-1,0]
            overlap_points_old = bins_avgr[i-1,:] <= bins_avgr[i,-1]
            curr_offset = -np.mean( dat_avgr[i,overlap_points_new] ) + np.mean( dat_avgr[i-1,overlap_points_old] ) + last_offset
        else:
            curr_offset = -np.mean( dat_avgr[i,-3:] )
        last_offset = curr_offset

        out_dat[i,:,0] = bins_avgr[i,:]
        out_dat[i,:,1] = dat_avgr[i,:]+curr_offset
        out_dat[i,:,2] = err_avgr[i,:]
        
    return out_dat

pos_list = np.array([20,60,100,150])
pos_sig_list = np.array([5.,5.,5.,5])  ## sigmas for position offset
def stitch_data( dat, is_cal = False ):
    ## take separate date returned by get_data and stitch it together into
    ## a single vector
    ## A systematic error given by the mean difference in the overlap regions is also included
    ## Also add the true positions estimated from the cantilever images

    out_dat = dat*1.0
    ## first step through and find all the syst errors
    ubins = np.unique( dat[:,:,0].flatten() )
    cbin = dat[:,:,0] 
    cdat = dat[:,:,1]
    cerr = dat[:,:,2]
    syst_err = []
    for u in ubins:
        cidx = cbin == u
        if( np.sum(cidx) <= 1 ): continue
        syst_err.append( np.std( cdat[cidx] )/np.sqrt( np.sum(cidx) ) )
    syst_err_over = np.percentile( np.abs(syst_err), 90 )
    print "Systematic error from overlap region is: ", syst_err_over
    
    ## now remake the data in a flattened vector at the appropriate bins with
    ## systematic error added
    if( True ): ##not is_cal):
        out_dat[:,:,2] = np.sqrt( dat[:,:,2]**2 + syst_err_over**2 )

    if( do_reposition ):
        #cant_pos = np.loadtxt( cant_pos_file, delimiter="," )
        print "Reposition bins using following positions: "
        print pos_offsets

        for i in range( np.shape(cbin)[0] ):
            min_bin = out_dat[i,0,0]
            min_bin_idx = np.argmin( np.abs(min_bin - pos_list) )
            delta_x = pos_offsets[i]
            print "for pos %f offset by %f" % (min_bin, delta_x)
            out_dat[i,:,0] = dat[i,:,0] + delta_x

    ## now turn into single array
    out_x = out_dat[:,:,0].flatten()
    out_y = out_dat[:,:,1].flatten()
    out_z = out_dat[:,:,2].flatten()

    sidx = np.argsort(out_x)

    return np.column_stack( (out_x[sidx], out_y[sidx], out_z[sidx]) )

def NLL(data, err, model):
    return np.sum( (data - model)**2/(2.0*err**2) )

def dipole_fun(x, Afix, A0):
    out_y = bu.get_es_force(x*1e-6,volt=Afix,is_fixed=True) + A0
    return out_y

def emp_dipole_fun(x, Afix, A0, es_dat):
    out_y = Afix * np.interp( x, es_dat[:,0], es_dat[:,1] ) + A0
    return out_y

def tot_fun(x, Afix, A0, beta):
    cf = bu.get_chameleon_force(x*1e-6)*beta
    #tot_out = (cf - np.min(cf) ) + emp_dipole_fun(x,Afix,A0,es_dat)
    tot_out = (cf - np.min(cf) ) + dipole_fun(x,Afix,A0)
    return tot_out

def offset_fun(xoff,x,p0):
    xmod = 1.0*x
    for i in range( np.shape(xmod)[0] ):
        min_bin = x[i,0,0]
        min_bin_idx = np.argmin( np.abs(min_bin - pos_list) )
        xmod[i,:,0] = x[i,:,0] + xoff[min_bin_idx]  
    bp_cal, bcov_cal = opt.curve_fit( dipole_fun, xmod[:,:,0].flatten(),x[:,:,1].flatten(),p0=p0)
    #bp_cal = p0
    sig_fix = 0.2*(x[:,:,1].flatten()) ## no errors for calibration so we need to weight the resids
    sig_meas = np.sqrt( x[:,:,2].flatten()**2 + sig_fix**2)
    chi2 = np.sum( (x[:,:,1].flatten() - dipole_fun(xmod[:,:,0].flatten(),*bp_cal))**2/sig_meas**2 )
    ## now also add gaussian constraints
    chi2 += np.sum( xoff**2/pos_sig_list**2 )

    #print xoff
    if(False):
        plt.close('all')
        plt.figure()
        plt.plot( x[:,:,0].flatten(), x[:,:,1].flatten(), 'bs' )
        plt.plot( xmod[:,:,0].flatten(), x[:,:,1].flatten(), 'ks' )
        plt.plot( xmod[:,:,0].flatten(), dipole_fun(xmod[:,:,0].flatten(),*bp_cal), 'ro' )
        plt.title( str(xoff) )
        print bp_cal
        plt.show()

    return chi2

def make_profile(data, beta_list, es_dat):
    ## step over each value of beta in beta list and calculate the best fit and likelihood
    profile = np.zeros_like(beta_list)
    for i,b in enumerate(beta_list):
        
        cfun = lambda x,Afix,A0: tot_fun(x,Afix,A0,b)
        spars = [1,0]
        try:
            bp,bcov = opt.curve_fit(cfun, data[:,0], data[:,1], sigma=data[:,2], p0=spars)
        except RuntimeError:
            bp = [0, 0]
        
        profile[i] = NLL( data[:,1], data[:,2], cfun(data[:,0],*bp) )

        if(False):
            print bp
            plt.close('all')
            plt.figure()
            plt.errorbar( data[:,0], data[:,1], yerr=data[:,2], fmt='ko' )
            plt.plot( data[:,0], cfun(data[:,0],*bp), 'r', label="total")
            plt.plot( data[:,0], dipole_fun(data[:,0],*bp), 'g--', label="electrostatic")
            plt.plot( data[:,0], tot_fun(data[:,0],0,0,b), 'b:', label="chameleon")            

            plt.title("beta = %e, NLL = %f"%(b,profile[i]) )
            plt.legend(loc="upper right")
            plt.show()

    return 2.0*(profile-np.min(profile))

xx = np.linspace(20, 250, 1e3)
xlims = [20, 230]
collist = ['b','g','r']

ms = 3.

pdf_out = PdfPages("plots/bead_fits.pdf")

prof_fig = plt.figure()
beta_list = np.linspace(0,5e5,30)
tot_prof = np.zeros_like(beta_list)
tot_data_fig = plt.figure()
plt_idx = 0
tdf_gs = gridspec.GridSpec(3,1,height_ratios=[1,1,1])
tdf_gs.update(hspace=0)

total_dat = []

for col,bd in zip(collist,bead_dicts):

    cal_idx = bd['cal_idx']
    cham_idx = bd['cham_idx']
    dc_volt_to_use = bd['dc_volt_to_use']
    cant_pos_file = bd['cant_pos_file']
    bead_label = bd['bead_label']

    cal_dat = get_data(cal_idx,is_cal=True)
    bp_cal_init, bcov_cal_init = opt.curve_fit( dipole_fun, cal_dat[:,:,0].flatten(),cal_dat[:,:,1].flatten())
    best_off = opt.minimize( offset_fun, [-1,-1,-1,-1], args=(cal_dat,bp_cal_init), method="Nelder-Mead" )
    #pos_offsets = [0., 0., 0., 0.,]
    pos_offsets = best_off.x
    print pos_offsets

    cal_dat = stitch_data(cal_dat,is_cal=True)
    spars = [357,0,-1e-15]
    bp_cal, bcov_cal = opt.curve_fit( dipole_fun, cal_dat[:,0],cal_dat[:,1])
    print "Best fit to cal [Afix, Aind, Ao]: ", bp_cal

    data_plot = plt.figure()
    plt.title("Force vs. position, " + bead_label)
    gs = gridspec.GridSpec(2,1,height_ratios=[1,1])
    plt.subplot(gs[0])
    plt.errorbar( cal_dat[:,0], cal_dat[:,1], yerr=cal_dat[:,2], fmt='bs', label="$V$ = 4V", markersize=ms)
    plt.plot(xx, dipole_fun(xx,*bp_cal), color=[0.25,0.5,1], linewidth=1.5, label="FE calc")

    #err_vals = cal_dat[:,1].flatten() - dipole_fun(cal_dat[:,0].flatten(),*bp_cal)
    #plt.errorbar( cal_dat[:,0].flatten(), err_vals, yerr=cal_dat[:,2].flatten(), fmt='ks')

    cham_dat = get_data(cham_idx,is_cal=False)
    cham_dat = stitch_data(cham_dat,is_cal=False)

    plt.errorbar( cham_dat[:,0], cham_dat[:,1], yerr=np.sqrt(cham_dat[:,2]**2), fmt='ko', label="$V$ = 0V", markersize=ms, linewidth=1.5)
    plt.ylabel("Force [N]")
    plt.xlim(xlims)
    plt.legend(loc="upper right", numpoints=1)

    plt.subplot(gs[1])
    plt.errorbar( cham_dat[:,0], cham_dat[:,1], yerr=np.sqrt(cham_dat[:,2]**2), fmt='ko', label="$V$ = 0V", markersize=ms, linewidth=1.5)
    
    profile = make_profile( cham_dat, beta_list, cal_dat)
    tot_prof = tot_prof + 1.0*profile

    total_dat.append( [cham_dat, cal_dat] )

    conf_lev = 2*1.35 ## NLL at 90% CL
    plt.figure(prof_fig.number)
    plt.plot( beta_list, profile, col+'-', linewidth=1.5, label=bead_label)

    ## get 90% CL and overplot on data
    cl90 = np.interp( conf_lev, profile, beta_list )

    plt.figure( data_plot.number )
    #plt.subplot(gs[1])
    cham_func = bu.get_chameleon_force(xx*1e-6)*cl90
    
    plt.plot(xx, cham_func-np.min(cham_func), 'r', label="90% UL", linewidth=1.5)    
    plt.legend(numpoints=1,loc=0)
    plt.xlabel("Distance from attractor [$\mu$m]")
    plt.xlim(xlims)
    plt.ylabel("Force [N]")

    pdf_out.savefig()

    plt.figure(tot_data_fig.number)
    plt.subplot( tdf_gs[plt_idx] )
    unit_scale = 1e15
    plt.errorbar( cham_dat[:,0], cham_dat[:,1]*unit_scale, yerr=np.sqrt(cham_dat[:,2]**2)*unit_scale, fmt='ko', markersize=ms, linewidth=1.5)
    spars = [357,0,-1e-15]
    bp_cham, bcov_cham = opt.curve_fit( dipole_fun, cham_dat[:,0],cham_dat[:,1],sigma=cham_dat[:,2])
    plt.plot(xx, dipole_fun(xx,*bp_cham)*unit_scale, '--', color=[0.25,0.5,1], linewidth=1.5)
    plt.plot(xx, (cham_func-np.min(cham_func))*unit_scale, 'r', linewidth=1.5)       

    plt.ylim([-0.1,0.2])
    plt.xlim([0,250])
    if(plt_idx == 1):
        plt.ylabel( "Force [fN]" )
    ax = plt.gca()
    if( plt_idx == 0 ):
        ax.set_yticks([-0.1, 0, 0.1, 0.2])
    else:
        ax.set_yticks([-0.1, 0, 0.1])
    if( plt_idx < 2 ):
        ax.set_xticklabels([])
    else:
        plt.xlabel("Distance [$\mu$m]")

    plt_idx += 1
    #plt.show()

plt.figure(prof_fig.number)
tot_prof -= np.min(tot_prof)
plt.plot( beta_list, tot_prof, 'k-', linewidth=2.5)
plt.plot( beta_list, conf_lev*np.ones_like(beta_list), ':',color=[0.5,0.5,0.5], linewidth=1.5, label="Combined")
ax = plt.gca()
ax.text(10000, conf_lev+0.05, r"90% CL",horizontalalignment="left",verticalalignment='bottom',color=[0.5,0.5,0.5],fontsize=14)
plt.legend(loc=0)
plt.xlabel(r"Chameleon matter coupling, $\beta$")
plt.ylabel(r"-2$\Delta$NLL")
plt.xlim([0,np.max(beta_list)])
plt.ylim([0,5])

plt.savefig("plots/profiles.pdf")

pdf_out.close()

plt.figure(tot_data_fig.number)
tot_data_fig.set_size_inches(5,4)
plt.subplots_adjust(top=0.97,right=0.96,left=0.14,bottom=0.115)
plt.savefig("plots/bead_data.pdf")

out_file = open("final_data_to_fit_for_paper.pkl", "wb")
pickle.dump(total_dat, out_file)
out_file.close()

plt.show()

