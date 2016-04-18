import bead_util as bu
import numpy as np
import matplotlib.pyplot as plt

## Constants
rb = 2.5e-6 ## 1/m
rhob = 2000. ## kg/m^3
mb = 4./3*np.pi*rb**3 * rhob
mpl = 3.6e-9 ## kg

## Background potential

xx = np.linspace(20,230,1e3)
cham_pot = np.loadtxt("cham_pot_1D.txt",skiprows=8)

def pot_weak_perturb( cham_pot, bead_pos, beta, npts=100 ):
    ## return perturbation to potential within the bead
    r = np.linspace(-rb, rb, npts)
    pot_pert = -mb*beta/(8*np.pi*rb*mpl)*(3 - r**2/rb**2)
    xpos = bead_pos + r
    tot_pot = np.interp(xpos, cham_pot[:,0], cham_pot[:,1]) + pot_pert
    #tot_pot = pot_pert
    return r, tot_pot

def max_weak_beta(phi_bg):
    ## find the max value of beta for the weak perturbation regime
    return 4*np.pi*rb*mpl*phi_bg/mb * 0.5

def equilib_pot( rho, Gamma, beta ):
    return np.sqrt( Gamma**5 * mpl/(beta * rho) )

def pot_strong_perturb( cham_pot, bead_pos, Gamma, beta, npts=100 ):
    r = np.linspace(-rb, rb, npts)
    phi_bg = np.interp( bead_pos+r, cham_pot[:,0], cham_pot[:,1] )
    phi_b = equilib_pot( rhob, Gamma, beta )
    s = rb*np.sqrt( 1 - 8*np.pi/3. * mpl/(beta*mb)*rb*phi_bg )
    
    print phi_b
    tot_pot = phi_b + 1./(8*np.pi*rb)*(beta*mb/mpl)*(np.abs(r)**3 -3*np.abs(r)*s**2 + 2*s**3)/(np.abs(r)*rb**2)
    tot_pot[ np.abs(r) < s ] = phi_b

    return r, tot_pot

bead_pos = 20e-6
phi_bg = np.interp(bead_pos,cham_pot[:,0], cham_pot[:,1])

## weak pert
beta = np.linspace(1,max_weak_beta(phi_bg),5)
plt.figure()
for b in beta:
    x,p = pot_weak_perturb( cham_pot, bead_pos, b)

    plt.plot(x,p,label="%e"%b)

## strong pert
beta = np.linspace(max_weak_beta(phi_bg)*2, max_weak_beta(phi_bg)*10,5)
for b in beta:
    x,p = pot_strong_perturb( cham_pot, bead_pos, 11700, b)

    plt.semilogy(x,p,'--',label="%e"%b)

#plt.legend(loc)
plt.show()
