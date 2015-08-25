## measure the force from the cantilever
import glob, os
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp

data_dir = "/data/20141212/Bead1"

dlist = ["cant_charge0","cant_charge1","cant_charge2",
         "cant_charge3","cant_charge4","cant_charge5"]
ncharges = ["+100", "+10", "+1", "0", "-10", "-100"]
clist = ["k", "r", "b", "g", "c", "m", "y"]

pos_list = ["-inf", "-5_5", "-5_0", "-4_5", "-4_0", "-3_5", "-3_0", "-2_5", "-3_5_v2", "-5_5_v2", "-inf_v2"]
pos_vals = np.array([-10, -5.5, -5., -4.5, -4, -3.5, -3, -2.5, -3.5, -5.5, -10])

def qfun( d, p ):
    return 1.0*p/d**2
def cfun( d, p ):
    return 1.0*p/d**3
def qcfun(d, p1, p2):
    return 1.0*p1/d**2 + 1.0*p2/d**3

xfig = plt.figure()
yfig = plt.figure()
zfig = plt.figure()
cal = True
for d,n,c in zip(dlist,ncharges,clist):

    cdir = os.path.join(data_dir, d)

    dvals = []
    for p,pv in zip(pos_list, pos_vals):
        
        cfile = os.path.join( cdir, "35mbar_cant_" + p + "_50mV_307Hz.h5" )
        if( not os.path.isfile(cfile)):
            cfile = os.path.join( cdir, "35mbar_cant_" + p + "in_50mV_307Hz.h5" )
        if( not os.path.isfile(cfile)):
            cfile = os.path.join( cdir, "35mbar_cant_" + p + "_100mV_307Hz.h5" )
        if( not os.path.isfile(cfile)):
            continue

        cdat = bu.getdata( cfile )[0]

        xdat, ydat, zdat = cdat[:,bu.data_columns[0]], cdat[:,bu.data_columns[1]], cdat[:,bu.data_columns[2]]

        ## first get the calibration factor
        if( cal ):
            sfac_x, bpx, _ = bu.get_calibration(cfile, [1, 1000],
                                            make_plot=False,
                                            data_columns = [0,1] )
            sfac_y, bpy, _ = bu.get_calibration(cfile, [1, 1000],
                                            make_plot=False,
                                            data_columns = [1,1] )
            sfac_z, bpz, _ = bu.get_calibration(cfile, [1, 1000],
                                            make_plot=False,
                                            data_columns = [2,1] )
            cal = False

            bpx[1], bpy[1], bpz[1] = 150, 150, 80
            kx, ky, kz = (2*np.pi*bpx[1])**2*bu.bead_mass, (2*np.pi*bpy[1])**2*bu.bead_mass, (2*np.pi*bpz[1])**2*bu.bead_mass
            print "k: ", kx, ky, kz


            sfac_x *= kx*1e12
            sfac_y *= ky*1e12
            sfac_z *= kz*1e12

        xdat *= abs(sfac_x)
        ydat *= abs(sfac_y)
        zdat *= abs(sfac_z)
        # plt.figure()
        # plt.plot(xdat)
        # plt.plot(ydat)
        # plt.plot(zdat)
        # plt.show()

        sN = np.sqrt(len(xdat))
        cdist = np.sqrt(pv**2 + 1.5**2 + 1.75**2)/(0.125)*5
        dvals.append([cdist, np.median(xdat), np.std(xdat)/sN,
                      np.median(ydat), np.std(ydat)/sN,
                      np.median(zdat), np.std(zdat)/sN])

        
    dvals = np.array(dvals)

    ## find infinity values
    inf_pts = dvals[:,0] > 350
    if( np.sum(inf_pts) == 0):
        inf_x, inf_y, inf_z = 2.5*sfac_x, 5.4*sfac_y, 5.0*sfac_z
    else:
        inf_x, inf_y, inf_z = np.mean(dvals[inf_pts,1]), np.mean(dvals[inf_pts,3]), np.mean(dvals[inf_pts,5])

    print inf_x, inf_y, inf_z

    plt.figure(xfig.number)
    plt.errorbar(dvals[:,0], dvals[:,1]-inf_x, yerr=0.15*sfac_x, fmt=c+'o', label="q="+n)
    plt.title("X position")
    plt.legend(loc="upper left")
    px2,_ = sp.curve_fit(qfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,1]-inf_x), p0=-1e4)
    px3,_ = sp.curve_fit(cfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,1]-inf_x), p0=-1e6)
    px4,_ = sp.curve_fit(qcfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,1]-inf_x), p0=[-1e-4,-1e6])
    xx = np.linspace( np.min(dvals[:,0]), np.max(dvals[:,0]), 1e2)
    plt.plot(xx, qfun(xx, px2), c+"--")
    plt.plot(xx, cfun(xx, px3), c+"-")
    plt.plot(xx, qcfun(xx, px4[0], px4[1]), c+":")
    plt.xlabel("3D distance [um]")
    plt.ylabel("Force [pN]")

    plt.figure(yfig.number)
    plt.errorbar(dvals[:,0], dvals[:,3]-inf_y, yerr=0.5*sfac_y, fmt=c+'o', label="q="+n)
    plt.title("Y position")
    plt.legend(loc="upper right")
    px2,_ = sp.curve_fit(qfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,3]-inf_y), p0=1e4)
    px3,_ = sp.curve_fit(cfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,3]-inf_y), p0=1e6)
    px4,_ = sp.curve_fit(qcfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,3]-inf_y), p0=[1e-4,1e6])
    xx = np.linspace( np.min(dvals[:,0]), np.max(dvals[:,0]), 1e2)
    plt.plot(xx, qfun(xx, px2), c+"--")
    plt.plot(xx, cfun(xx, px3), c+"-")
    plt.plot(xx, qcfun(xx, px4[0], px4[1]), c+":")
    plt.xlabel("3D distance [um]")
    plt.ylabel("Force [pN]")

    plt.figure(zfig.number)
    plt.errorbar(dvals[:,0], dvals[:,5]-inf_z, yerr=0.2*sfac_z, fmt=c+'o', label="q="+n)
    plt.title("Z position")
    plt.legend(loc="upper right")
    px2,_ = sp.curve_fit(qfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,5]-inf_z), p0=-1e4)
    px3,_ = sp.curve_fit(cfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,5]-inf_z), p0=-1e6)
    px4,_ = sp.curve_fit(qcfun, np.transpose(dvals[:,0]), np.transpose(dvals[:,5]-inf_z), p0=[-1e-4,-1e6])
    xx = np.linspace( np.min(dvals[:,0]), np.max(dvals[:,0]), 1e2)
    plt.plot(xx, qfun(xx, px2), c+"--")
    plt.plot(xx, cfun(xx, px3), c+"-")
    plt.plot(xx, qcfun(xx, px4[0], px4[1]), c+":")
    plt.xlabel("3D distance [um]")
    plt.ylabel("Force [pN]")


plt.show()

