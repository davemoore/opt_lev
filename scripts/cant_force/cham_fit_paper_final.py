import glob, os, re, sys
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cPickle as pickle
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy.interpolate as interp

def NLL(data, err, model, constr):
    ## Asig constraint error
    #gc_err = 0 
    #for c in constr:
    #    gc_err += ((c[0]-1)/(2*c[1]))**2
    return np.sum( (data - model)**2/(2.0*err**2) ) #+ gc_err

# def dipole_fun(x, Afix, A0):
#     out_y = bu.get_es_force(x*1e-6,volt=Afix,is_fixed=True) + A0
#     return out_y

def emp_dipole_fun(x, Afix, A0):
    ifn = interp.UnivariateSpline(es_dat[:,0], es_dat[:,1],s=1e-31)
    # plt.figure()
    # plt.plot( es_dat[:,0], es_dat[:,1], 'ko')
    # plt.plot( 0.9*x, ifn(0.9*x), 'r')
    # plt.show()
    #out_y = np.abs(Afix) * np.interp( x, es_dat[:,0], es_dat[:,1] ) + A0
    out_y = np.abs(Afix) * ifn( x ) + A0
    return out_y

def tot_fun(x, Afix, A0, beta, Asig):
    #cf = bu.get_chameleon_force(x*1e-6)*beta
    #tot_out = (cf - np.min(cf) ) + emp_dipole_fun(x,Afix,A0,es_dat)
    #cf = bu.get_cham_vs_beta_x_lam(x*1e-6, 1e6, lam)*beta/1e6

    cf = Asig*1e16*bu.get_cham_vs_beta_x_lam(x*1e-6, beta, lam)
    tot_out = (cf - np.min(cf) ) + emp_dipole_fun(x,Afix,A0)
    return tot_out

def make_profile(data, beta_list, es_dat):
    ## step over each value of beta in beta list and calculate the best fit and likelihood
    profile = np.zeros_like(beta_list)
    for i,b in enumerate(beta_list):
        
        cfun = lambda x,Afix,A0,Asig: np.hstack((tot_fun(x[:-1],Afix,A0,b,Asig),[Asig,]))
        spars = [0.001,0,1e-16]
        try:
            #bp,bcov = opt.curve_fit(cfun, data[:,0], data[:,1], sigma=data[:,2], p0=spars)
            xd = np.hstack((data[:,0], [1,]))
            yd = np.hstack((data[:,1], [1*1e-16,]))
            sd = np.hstack((data[:,2], [0.36*1e-16,]))
            bp,bcov = opt.curve_fit(cfun, xd, yd, sigma=sd, p0=spars)
        except RuntimeError:
            bp = [0, 0, 0]
        
        constr = [[bp[-1], 0.36],]
        #profile[i] = NLL( data[:,1], data[:,2], cfun(data[:,0],*bp), constr )
        profile[i] = NLL( yd, sd, cfun(xd,*bp), constr )

        if( profile[i] > conf_lev * 20 ): 
            profile[i:] = 1.0*profile[i]
            break

        if(False):
            print bp
            plt.close('all')
            plt.figure()
            plt.errorbar( data[:,0], data[:,1], yerr=data[:,2], fmt='ko' )
            plt.plot( data[:,0], cfun(data[:,0],*bp), 'r', label="total")
            plt.plot( data[:,0], emp_dipole_fun(data[:,0],*bp), 'g--', label="electrostatic")
            plt.plot( data[:,0], tot_fun(data[:,0],0,0,b), 'b:', label="chameleon")            

            plt.title("beta = %e, NLL = %f"%(b,profile[i]) )
            plt.legend(loc="upper right")
            plt.show()

    return 2.0*(profile-np.min(profile))



out_file = open("final_data_to_fit_for_paper.pkl", "rb")
bead_data = pickle.load( out_file )
out_file.close()

xx = np.linspace(20, 250, 1e3)
xlims = [20, 230]
ms = 3.
conf_lev = 2*1.35 ## NLL at 90% CL
collist = ['b','g','r']
bead_label_list = [r"$\mu$Sphere 1", r"$\mu$Sphere 2", r"$\mu$Sphere 3"]
lam_list = [4.6,5,5.5,6,6.5,7,12,30,60,100]
UL_list = []
for lam in lam_list:
    print "Working on lambda: ", lam
    
    prof_fig = plt.figure()
    tot_data_fig = plt.figure()
    plt_idx = 0
    tdf_gs = gridspec.GridSpec(3,1,height_ratios=[1,1,1])
    tdf_gs.update(hspace=0)

    beta_max = bu.beta_max(lam)
    down_fac = 1.0
    beta_list = np.linspace(1,3e5*(4.5/lam)**2,100)
    tot_prof = np.zeros_like(beta_list)

    for bead_label,col,data in zip(bead_label_list,collist,bead_data):

        cham_dat, cal_dat = data[0], data[1]
        es_dat = cal_dat

        profile = make_profile( cham_dat, beta_list, cal_dat)
        tot_prof = tot_prof + 1.0*profile
        plt.figure(prof_fig.number)
        plt.plot( beta_list, profile, col+'-', linewidth=1.5, label=bead_label)
        ## get 90% CL and overplot on data
        if( profile[-1] < conf_lev ):
            cl90 = 1e12
        else:
            cl90 = np.interp( conf_lev, profile, beta_list )
            
        cham_func = bu.get_cham_vs_beta_x_lam(xx*1e-6, cl90, lam)

        plt.figure(tot_data_fig.number)
        plt.subplot( tdf_gs[plt_idx] )
        unit_scale = 1e15
        plt.errorbar( cham_dat[:,0], cham_dat[:,1]*unit_scale, yerr=np.sqrt(cham_dat[:,2]**2)*unit_scale, fmt='ko', markersize=ms, linewidth=1.5)
        spars = [357,0,-1e-15]
        bp_cham, bcov_cham = opt.curve_fit( emp_dipole_fun, cham_dat[:,0],cham_dat[:,1],sigma=cham_dat[:,2])
        plt.plot(xx, emp_dipole_fun(xx,*bp_cham)*unit_scale, '--', color=[0.25,0.5,1], linewidth=1.5)
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

    plt.figure(tot_data_fig.number)
    tot_data_fig.set_size_inches(5,4)
    plt.subplots_adjust(top=0.97,right=0.96,left=0.14,bottom=0.115)
    plt.savefig("plots/bead_data_%d.pdf"%lam)

    plt.figure(prof_fig.number)
    tot_prof -= np.min(tot_prof)
    plt.plot( beta_list, tot_prof, 'k-', linewidth=2.5)
    plt.plot( beta_list, conf_lev*np.ones_like(beta_list), ':',color=[0.5,0.5,0.5], linewidth=1.5, label="Combined")
    ax = plt.gca()
    ax.text(0.05*np.max(beta_list), conf_lev+0.05, r"90% CL",horizontalalignment="left",verticalalignment='bottom',color=[0.5,0.5,0.5],fontsize=14)
    plt.legend(loc=0)
    plt.xlabel(r"Chameleon matter coupling, $\beta$")
    plt.ylabel(r"-2$\Delta$NLL")
    plt.title(r"$\Lambda$ = %f"%lam)
    plt.xlim([0,np.max(beta_list)])
    plt.ylim([0,5])
    plt.savefig("plots/profiles_%d.pdf"%lam)

    plt.close(prof_fig)
    plt.close(tot_data_fig)

    if( tot_prof[-1] < conf_lev ):
        cl90 = 1e12
    else:
        cl90 = np.interp( conf_lev, tot_prof, beta_list )
    UL_list.append([lam, cl90, beta_max])

UL_list = np.array(UL_list)
np.save("final_cham_limits_paper.npy", UL_list)
for i in range( np.shape(UL_list)[0] ):
    print "Lambda = %d:  UL=%e, MaxBeta=%e" % (UL_list[i,0], UL_list[i,1], UL_list[i,2])

plt.show()
