import numpy as np
import matplotlib.pyplot as plt

space = 0.5

#lam = 2e-4 ## length scale in m
hbar = 4.14e-15 ## eV s
c = 3e8 ## m/s

cdat = np.loadtxt("/home/dcmoore/comsol/dark_photons/elec_field_%smm_gap_1V_double_sharp.txt"%str(space), skiprows=9)

epsilon0 = 8.85e-12 ## F/m
d = 1e-3*space ## parallel plate sep in m

x = np.reshape(cdat[:,0], (1000, 1000))
y = np.reshape(cdat[:,1], (1000, 1000))
Ex = np.reshape(cdat[:,2], (1000, 1000))
Ey = np.reshape(cdat[:,3], (1000, 1000))
En = np.reshape(cdat[:,4], (1000, 1000))

dx, dy = 0.02/1000, 0.02/1000
px = x-dx/2
py = y-dy/2

#plt.pcolormesh(px,py,Ey)
#plt.show()
shield_pos = 0.0
elec_pos = -0.002

yidx_shield = np.argmin( np.abs( y[:,0] - shield_pos ) )
yidx_elec = np.argmin( np.abs( y[:,0] - elec_pos ) )

shield_map = np.zeros((500, 3))
for i,cx in enumerate(x[0,500:]):
    
    cdat = En[:,500+i]

    max_pos = np.argmax( -np.diff(cdat) )
    ypos = y[max_pos,0]
    maxx = -np.max( -np.diff(cdat) )

    shield_map[i,:] = [cx, ypos, maxx*epsilon0]
    # print shield_map[i,:]

    # plt.figure()
    # plt.plot( x[0,:-1], -np.diff(cdat) )
    # plt.show()

grad_shield = (Ey[yidx_shield+1,:]-Ey[yidx_shield,:])*epsilon0
grad_elec = (Ey[yidx_elec+1,:]-Ey[yidx_elec,:])*epsilon0



# plt.figure()
# plt.plot( y[:,0], grad_shield )
# plt.plot( y[:,0], grad_elec )
# plt.show()

def integrand( z, rvals, sigr, lam ):
    integ = sigr/(2*epsilon0) * rvals*z/(rvals**2 + z**2)**1.5 * (1 + 1./lam * np.sqrt( rvals**2 + z**2 )) * np.exp( -np.sqrt(rvals**2 + z**2)/lam )

    #zelec = z + d
    #integ2 = sigs/(2*epsilon0) * rvals*zelec/(rvals**2 + zelec**2)**1.5 * (1 + 1./lam * np.sqrt( rvals**2 + zelec**2 )) * np.exp( -np.sqrt(rvals**2 + zelec**2)/lam )

    return rvals, integ #+integ2

zarr = np.linspace(0, 0.001, 1e2)

earr_pp = np.zeros_like(zarr)
print "Starting integration..."
#massarr = np.logspace(-6, -1, 1e2)
massarr = np.logspace(-10,0,1e2)
ztouse = np.linspace( 0.000010, 0.000060, 1e2)

yy = np.linspace(0,0.005,1e4)
#evals = np.interp( yy, y[:,0], grad_shield )
evals = np.interp( yy, shield_map[:,0], shield_map[:,2] )
xvals = np.interp( yy, shield_map[:,0], shield_map[:,1] )

alpha = 5.8e-22 ## F m^2 (for Ba titanate)
Edc = 1e6 ##V/m
V = 2e3/1e-3 * d ## V
sigF = 4e-21
q = 5e5 * 1.6e-19 ## C

chiarr = np.zeros_like(massarr)
chiarr2 = np.zeros_like(massarr)

for i,mass in enumerate(massarr):

    lam = hbar*c/mass
    #print lam

    earr = np.zeros_like(ztouse)    
    for j, z in enumerate(ztouse):
        r, v = integrand(z+xvals, yy, evals, lam )
        earr[j] = np.trapz(v, r) * (1. - np.exp(-d/lam) )
    dEdy = np.gradient(earr, np.gradient(ztouse))
    dEdy_cent = np.median(dEdy)

    # plt.figure()
    # plt.plot( ztouse, earr )
    # plt.show()

    chiarr[i] = np.sqrt(sigF/(alpha*Edc*dEdy_cent*V))
    chiarr2[i] = np.sqrt(sigF/(np.abs(np.median(earr))*V*q))


np.save("data/dphot_lim_%s.npy"%str(space), np.vstack( (massarr, chiarr, chiarr2) ))

plt.figure()
plt.loglog( massarr, chiarr )
plt.loglog( massarr, chiarr2 )
plt.show()
