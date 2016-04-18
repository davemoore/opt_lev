import math, sys
from scipy import integrate
import numpy as np
#import matplotlib.pyplot as plt

lam = float(sys.argv[1])

zoff_list = np.linspace(20,230,50)*1e-6

## calculate the yukawa force over a distributed test mass assumed to be cube

D = 5e-6 # diameter of bead (m)
rhob = 2e3 # density bead (kg/m^3)
rhoa = 2.3e3 # density attractor
a_thick = 10e-6 # length of attractor cube side (m)
a_depth = 200e-6 # depth of attractor cube side (m)
a_width = 200e-6

rb = D/2.0

def dV(phi,theta,r):
    return r**2 * math.sin(theta)

alpha = 1e15
G = 6.67398e-11 

def Vg_tot(currx,curry,currz):
    d = np.sqrt( currx**2 + curry**2 + currz**2 )
    Vout = alpha*lam/d*( np.exp(-(d + rb)/lam)*( lam**2 + lam*rb + rb**2) 
                                               - np.exp(-(d - rb)/lam)*( lam**2 - lam*rb + rb**2) )
    return Vout

fix_term = alpha*np.exp(-rb/lam)*( (lam**2 + rb**2) * (np.exp(2*rb/lam) -1) - lam*rb*(np.exp(2*rb/lam)) )
print fix_term

def Fz_tot(currx,curry,currz):
    x = currx + rb + a_depth/2.0 + zoff
    y = curry
    z = currz
    d = np.sqrt( x**2 + y**2 + z**2 )
    Fzout = fix_term*x*(lam+d)/d**3 * np.exp(-d/lam)
    return Fzout

curr_thick = a_depth

force_list = []
for zoff in zoff_list:
    intval = integrate.tplquad(Fz_tot, -a_depth/2.0, a_depth/2.0, lambda y: -a_thick/2.0, lambda y: a_thick/2.0, lambda y,z: -a_width/2.0, lambda y,z: a_width/2.0, epsrel=1e-2 )

    #print intval

    integ = intval[0] * -2.*np.pi*G*rhob*rhoa/alpha
    integ_err = intval[1] * -2.*np.pi*G*rhob*rhoa/alpha

    force_list.append( [zoff, integ, integ_err] )

    print "integral is: ", zoff, integ, integ_err

force_list = np.array(force_list)

fname = 'cham_data/force_lam_%.3f.npy' % (lam*1e6 )
np.save( fname, force_list )

# plt.figure()
# plt.plot( force_list[:,0], force_list[:,1] )
# plt.show()



         
                        


