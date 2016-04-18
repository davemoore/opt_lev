import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import cPickle as pickle

data_dir = "/home/dcmoore/opt_lev/scripts/force_calc/cham_data"

lam_list = np.logspace(0,2.0,20)*1e-6

cdat0 = np.load( data_dir + "/force_lam_%.3f.npy" % (lam_list[0]*1e6 ) )

xvec = cdat0[:,0]
lamvec = lam_list*1.0
tot_dat = np.zeros( (len(lamvec), len(xvec)) )
for j,lam in enumerate(lam_list):

    cdat = np.load( data_dir + "/force_lam_%.3f.npy" % (lam*1e6 ) )
    tot_dat[j, :] = -cdat[:,1]

yuk_spl = interp.RectBivariateSpline(lamvec, xvec, tot_dat, s=0)

plt.figure()
for j,lam in enumerate(lam_list):

    plt.loglog(xvec, tot_dat[j,:])
    plt.loglog(xvec, yuk_spl(lam, xvec )[0], '.')
    
if(True):
    outfile = open("yuk_force_arr.pkl","wb")
    out_dat = [lamvec, xvec, tot_dat]
    pickle.dump(out_dat, outfile)
    outfile.close()

plt.show()
