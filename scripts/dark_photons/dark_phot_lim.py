import numpy as np
import matplotlib.pyplot as plt

space = 8

#lam = 2e-4 ## length scale in m
hbar = 4.14e-15 ## eV s
c = 3e8 ## m/s

#cdat = np.loadtxt("/home/dcmoore/comsol/dark_photons/elec_field_%smm_gap_1V_sharp.txt"%str(space), skiprows=9)
cdat = np.loadtxt("comsol/dark_photons/elec_field_2mm_gap_1V_sharp.txt", skiprows=9)

epsilon0 = 8.85e-12 ## F/m
d = 1e-3*space ## parallel plate sep in m

x = np.reshape(cdat[:,0], (1000, 1000))
y = np.reshape(cdat[:,1], (1000, 1000))
Ey = np.reshape(cdat[:,2], (1000, 1000))

dx, dy = 0.02/1000, 0.02/1000
px = x-dx/2
py = y-dy/2

plt.figure()
plt.pcolormesh(px,py,Ey)
plt.show()

shield_pos = 0.0
elec_pos = -0.002

yidx_shield = np.argmin( np.abs( y[:,0] - shield_pos ) )
yidx_elec = np.argmin( np.abs( y[:,0] - elec_pos ) )

grad_shield = (Ey[yidx_shield,:]-Ey[yidx_shield-1,:])*epsilon0
grad_elec = (Ey[yidx_elec,:]-Ey[yidx_elec-1,:])*epsilon0

# plt.figure()
# plt.plot( y[:,0], grad_shield )
# plt.plot( y[:,0], grad_elec )
# plt.show()

def integrand( z, rvals, sigr, lam ):
    
    #r = np.sqrt( (yface - ycurr)**2 + z**2 )
    #integ = sigr/(2*epsilon0) *  z/r**3 * (1 + r/lam) * np.exp(-r/lam)

    integ = sigr/(2*epsilon0) * rvals*z/(rvals**2 + z**2)**1.5 * (1 + 1./lam * np.sqrt( rvals**2 + z**2 )) * np.exp( -np.sqrt(rvals**2 + z**2)/lam )

    #zelec = z + d
    #integ2 = sigs/(2*epsilon0) * rvals*zelec/(rvals**2 + zelec**2)**1.5 * (1 + 1./lam * np.sqrt( rvals**2 + zelec**2 )) * np.exp( -np.sqrt(rvals**2 + zelec**2)/lam )

    return rvals, integ #+integ2

zarr = np.linspace(0, 0.001, 1e2)

earr_pp = np.zeros_like(zarr)
print "Starting integration..."

pxvals = 1.0*y[:,0]
pyvals = 1.0*x[0,:]
dp_field = np.zeros_like(Ey)

print len(pxvals)

#yy_face = np.linspace(0,0.010,5e3)
#evals = np.interp( yy_face, y[:,0], grad_shield )

#lam = hbar*c/(0.01)
#print lam
# for i,xx in enumerate(pxvals):
#     if( i % 100 == 0 ): print i
#     if( abs(xx/lam) > 5 ):
#         continue
#     if( xx < 0 ): continue

#     for j,yy in enumerate(pyvals):
        
#         if( yy < 0 ):
#              continue

#         r, v = integrand(np.abs(xx), yy_face, yy, evals, lam )
#         dp_field[i,j] = np.trapz(v, r) * (1. - np.exp(-d/lam) )**2

# plt.figure()
# plt.pcolormesh(px,py,np.abs(dp_field))
# plt.colorbar()
# plt.show()


#massarr = np.logspace(-6, -1, 1e2)
massarr = np.logspace(-7,2,2e2)
#ztouse = np.linspace( 0.000005, 0.000010, 1e2)
ztouse = np.linspace( 0.00000, 0.00010, 1e2)

yy = np.linspace(0,0.005,1e4)
evals = np.interp( yy, y[:,0], grad_shield )

alpha = 5e-24 ## F m^2 (for Ba titanate, 5 um diam)
Edc = 5e6 ##V/m
V = 5e3/1e-3 * d ## V
sigF = 2e-23
#sigF = 1e-23
q = 5e5 * 1.6e-19 ## C

chiarr = np.zeros_like(massarr)
chiarr2 = np.zeros_like(massarr)

massarr = [1e-2,]
for i,mass in enumerate(massarr):

    lam = hbar*c/mass
    #print lam

    earr = np.zeros_like(ztouse)    
    for j, z in enumerate(ztouse):
        r, v = integrand(z, yy, evals, lam )
        earr[j] = np.trapz(v, r) * (1. - np.exp(-d/lam) )**2

    plt.figure()
    plt.plot(ztouse, earr)
    plt.show()

    dEdy = np.gradient(earr, np.gradient(ztouse))
    dEdy_cent = np.median(dEdy)

    chiarr[i] = np.sqrt(sigF/(alpha*Edc*dEdy_cent*V))
    chiarr2[i] = np.sqrt(sigF/(np.abs(np.median(earr))*V*q))

print chiarr
print chiarr2
#np.save("data/dphot_lim_%s_smallbead.npy"%str(space), np.vstack( (massarr, chiarr, chiarr2) ))

plt.figure()
plt.loglog( massarr, chiarr )
plt.loglog( massarr, chiarr2, 'g' )
plt.show()
