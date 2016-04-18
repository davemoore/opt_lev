import bead_util as bu
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.interpolate as interp

lambda_list = [2.3, 5, 10, 50, 100]
beta_list = [1, 1e2, 1e3, 1e4, 1e5, 1e6,1e7,1e8]

rb = 2.5e-6 #m
rhob = 2000. #kg/m^3
mb = 4./3*np.pi*rb**3 * rhob #kg
mpl = 3.6e-9 ## kg (=2e18 GeV)


bnds_arr = np.zeros( (len(lambda_list),len(beta_list)) )
plt.figure()
xx = np.linspace(10e-6,500e-6,200)
cham_force_arr = np.zeros( (len(beta_list), len(xx), len(lambda_list)) )

# outfile = open("cham_force_arr.pkl","rb")
# out_dat = pickle.load(outfile)
# outfile.close()

# ## make an interpolator
# interp_fun = interp.RegularGridInterpolator(out_dat[:3],10.0**out_dat[3],bounds_error=False)

#print interp_fun([[1e6, 30e-6, 10],])
#raw_input('e')
 
plt.figure()
for j,lam in enumerate(lambda_list):
    
    lamfile = 0 if lam==2.3 else lam

    plt.title(str(lam))
    for k,beta in enumerate(beta_list):

        if( beta <= 1e6 ):
            betastr = "%.0e"%beta
        else:
            betastr = "1e6"
        betastr = betastr.replace("+0","")
        betastr = betastr.replace("1e0","1")
        cdat = np.loadtxt( "force_curves/potential_nobead_beta%s_lambda%d.txt"%(betastr,lamfile), skiprows = 8 )

        ## fix bad files
        if( lam == 50 and beta == 1e6 ):
            cdat[:,3] *= 10.
        if( lam == 5 and beta > 1):
            cdat[:,3] *= 2.36e11

        #plt.figure()
        ## force curves
        scale_fac = 1
        if( beta > 1e6 ): scale_fac = beta/1e6
        gpts = np.logical_and( cdat[:,0] >= 20e-6, cdat[:,0] <= 500e-6)
        #plt.loglog( cdat[:,0], cdat[:,3]*scale_fac, label="%e"%beta )
        #bvec = beta*np.ones_like(cdat[gpts,0])
        xvec = cdat[gpts,0]
        #lvec = lam*np.ones_like(cdat[gpts,0])*2
        #plt.loglog( cdat[gpts,0], bu.get_cham_vs_beta_x_lam(xvec,beta,lam), 'o' )

        pbg20 = np.interp(20e-6, cdat[:,0], cdat[:,1])
        clamsup = (mb/(4./3*np.pi*rb**3)*rb**2)/(3*mpl/beta*pbg20)  #4*np.pi*rb*mpl*pbg20/mb
        sval = rb*np.sqrt( 1 - 8*np.pi/3. * mpl/(beta*mb) * rb * pbg20 )
        lamval = 1 - (sval/rb)**3
        if( clamsup <= 0.75 ): lamval = 1.0

        cham_force_arr[k,:,j] = np.interp(xx, cdat[:,0], np.log10(cdat[:,3]*scale_fac*lamval))

        ## potentials
        phibg = cdat[:,1]
        cond = beta*mb/(4*np.pi*rb * mpl * phibg)
        gpts = cdat[:,0] > 20e-6
        #plt.plot( cdat[gpts,0], cond[gpts], label="%e"%beta )
        
        bnds_arr[j,k] = np.interp(20e-6,cdat[:,0],cond)

        # bb = np.logspace(0,10,1e3)
        # clamsup = (mb/(4./3*np.pi*rb**3)*rb**2)/(3*mpl/bb*pbg20)  #4*np.pi*rb*mpl*pbg20/mb
        # svec = rb*np.sqrt( 1 - 8*np.pi/3. * mpl/(bb*mb) * rb * pbg20 )
        # lamvec = 1 - (svec/rb)**3
        # lamvec[clamsup <= 0.75] = 1.
        # plt.figure()
        # plt.semilogx( bb, lamvec*bb )
        # yy = plt.ylim()
        # #plt.plot([clamsup, clamsup], yy, 'k')
        # plt.title(str(lam) + ", " + str(beta))
        # plt.show()

if(True):
    outfile = open("cham_force_arr.pkl","wb")
    out_dat = [beta_list, xx, lambda_list, cham_force_arr]
    pickle.dump(out_dat, outfile)
    outfile.close()

#plt.show()

plt.figure()
max_list = []
for j in range( np.shape(bnds_arr)[0] ):
    plt.loglog( beta_list, bnds_arr[j,:], label="%e"%lambda_list[j] )
    max_list.append( [lambda_list[j], np.interp(1.0, bnds_arr[j,:], beta_list)] )

xx = plt.xlim()
plt.plot( xx, [1,1], 'k--')

plt.legend(loc=0)

max_list = np.array(max_list)
plt.figure()
plt.loglog( max_list[:,0], max_list[:,1], 'ks' )
p = np.polyfit( max_list[:,0], max_list[:,1], 4 )
xx = np.linspace(max_list[:,0][0],max_list[:,0][-1],1e3)
#plt.plot(xx, np.polyval(p,xx), 'r')
plt.plot(xx, bu.beta_max(xx), 'r')


plt.show()
