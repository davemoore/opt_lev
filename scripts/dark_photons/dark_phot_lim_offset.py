import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.interpolate as interp
import numpy.ma as ma
space = 8

import matplotlib
matplotlib.rcParams['font.size'] = 15

#lam = 2e-4 ## length scale in m
hbar = 4.14e-15 ## eV s
c = 3e8 ## m/s

#cdat = np.loadtxt("/home/dcmoore/comsol/dark_photons/elec_field_%smm_gap_1V_sharp.txt"%str(space), skiprows=9)
cdat = np.loadtxt("/home/dcmoore/comsol/dark_photons/elec_field_2mm_gap_1V_sharp_memb.txt", skiprows=9)

epsilon0 = 8.85e-12 ## F/m
d = 1e-3*space ## parallel plate sep in m

x = np.reshape(cdat[:,0], (1000, 1000))
y = np.reshape(cdat[:,1], (1000, 1000))
Ey = np.reshape(cdat[:,2], (1000, 1000))

dx, dy = 0.02/1000, 0.02/1000
px = x-dx/2
py = y-dy/2

# plt.figure()
# plt.pcolormesh(px,py,Ey)
# plt.colorbar()
# plt.show()

shield_pos = 0.0
elec_pos = -0.002

yidx_shield = np.argmin( np.abs( y[:,0] - shield_pos ) )
yidx_elec = np.argmin( np.abs( y[:,0] - elec_pos ) )

grad_shield = (Ey[yidx_shield+2,:]-Ey[yidx_shield-2,:])
print "Max field: ", np.max(abs(grad_shield))
grad_elec = (Ey[yidx_elec,:]-Ey[yidx_elec-1,:])

# plt.figure()
# plt.plot( y[:,0], grad_shield )
# # plt.plot( y[:,0], grad_elec )
# plt.show()

theta_vec = np.linspace(0,2*np.pi,20)

xm = np.linspace(-0.0067,0.0067,100)
ym = np.linspace(-0.0067,0.0067,100)
xmesh, ymesh = np.meshgrid( xm, ym )
xd = np.median(np.diff(xm))
yd = np.median(np.diff(ym))

chargemesh = np.interp( ymesh, y[:,0], grad_shield )

def xyint( z, y, lam ):

    r = np.sqrt( z**2 + xmesh**2 + (ymesh-y)**2 )

    integ = np.sum( chargemesh * np.abs(z)/r**3 * (1+r/lam) * np.exp(-r/lam) )

    return integ/(2*np.pi) * xd*yd


def integrand( z, yface, ycurr, sigr, lam ):

    integ = np.zeros_like(yface)
    for ii,yf in enumerate(yface):

        r = np.sqrt( (yf*np.cos(theta_vec))**2 + (yf*np.sin(theta_vec) - ycurr)**2 + z**2 )
        integ[ii] = np.trapz(sigr[ii]*yf*z/r**3 * (1+r/lam) * np.exp(-r/lam),theta_vec)

    integ *= 1/(4*np.pi*epsilon0)
        #integ = sigr/(4*np.pi*epsilon0) *  z/r**3 * (1 + r/lam) * np.exp(-r/lam)

    #integ = sigr/(2*epsilon0) * rvals*z/(rvals**2 + z**2)**1.5 * (1 + 1./lam * np.sqrt( rvals**2 + z**2 )) * np.exp( -np.sqrt(rvals**2 + z**2)/lam )

    #zelec = z + d
    #integ2 = sigs/(2*epsilon0) * rvals*zelec/(rvals**2 + zelec**2)**1.5 * (1 + 1./lam * np.sqrt( rvals**2 + zelec**2 )) * np.exp( -np.sqrt(rvals**2 + zelec**2)/lam )

    return yface, integ #+integ2

zarr = np.linspace(0, 0.001, 1e2)

earr_pp = np.zeros_like(zarr)
print "Starting integration..."

pxvals = 1.0*y[:,0]
pyvals = 1.0*x[0,:]


print len(pxvals)

yy_face = np.linspace(0,0.0075,50)
evals = np.interp( yy_face, y[:,0], grad_shield )

lam = hbar*c/(0.01)
if(False):
    dp_field = np.zeros_like(Ey)
    print lam
    for i,xx in enumerate(pxvals):
        if( abs(xx/lam) > 8 ):
            continue
        if( i % 10 == 0 ): print i
        #if( xx < 0 ): continue

        for j,yy in enumerate(pyvals):

            #if( yy < 0 ):
            #     continue

            #r, v = integrand(np.abs(xx), yy_face, yy, evals, lam )
            #dp_field[i,j] = np.trapz(v, r) * (1. - np.exp(-d/lam) )**2
            dp_field[i,j] = xyint(xx,yy,lam) * (1. - np.exp(-d/lam) )**2

    of = open("dp_field_%d.pkl"%(lam*1e6), 'wb')
    pickle.dump([px,py,dp_field],of)
    of.close()
else:
    of = open("dp_field_%d.pkl"%(lam*1e6), 'rb')    
    cdat = pickle.load(of)
    of.close()
    dp_field=cdat[2]

    ## spline across each row
    for i in range(490,511):
        
        if( i < 495 ):
            sval = 0.5e3
        elif( i < 498 ):
            sval = 50e3
        elif( i < 501 ):        
            sval = 50e3
        elif( i < 505):
            sval = 50e3
        else:
            sval = 0.5e3
        spl = interp.UnivariateSpline(1.0*x[i,::7], 1.0*dp_field[i,::7],s=sval)
        spts = spl(x[i,:])*1.0
        # plt.figure()
        # plt.plot(x[i,:], dp_field[i,:])
        # plt.plot(x[i,:], spts, 'r')
        gpts = np.abs(dp_field[i,:]) > 10
        dp_field[i,gpts] = spts[gpts]
        # plt.plot(x[i,:], dp_field[i,:],'g')
        # plt.title(str(i))
        # plt.show()

dp_field = ma.array(Ey-dp_field, mask = Ey-dp_field==0)

def draw_filleted(w,h,r,sx,sy):
    lc = [1,1,1]
    lw = 1.5
    if( w < 0 ):
        plt.plot( [sx, sx + (w+r)], [sy, sy], color=lc, lw=lw )
        theta = np.linspace(-3*np.pi/2,-np.pi,1e2)
        xx = r*np.cos(theta)
        yy = r*np.sin(theta)
        plt.plot( xx + sx + (w+r), sy+r-yy, color=lc, lw=lw )
        plt.plot( [sx + w, sx+w], [sy + r, sy+h-r], color=lc, lw=lw )
        plt.plot( xx + sx + (w+r), sy+h-r+yy, color=lc, lw=lw )
        plt.plot( [sx, sx + (w+r)], [sy+h, sy+h], color=lc, lw=lw )
    else:
        plt.plot( [sx, sx + (w-r)], [sy, sy], color=lc, lw=lw )
        theta = np.linspace(-3*np.pi/2,-np.pi,1e2)
        xx = r*np.cos(theta)
        yy = r*np.sin(theta)
        plt.plot( -xx + sx + (w-r), sy+r-yy, color=lc, lw=lw )
        plt.plot( [sx + w, sx+w], [sy + r, sy+h-r], color=lc, lw=lw )
        plt.plot( -xx + sx + (w-r), sy+h-r+yy, color=lc, lw=lw )
        plt.plot( [sx, sx + (w-r)], [sy+h, sy+h], color=lc, lw=lw )    

fig=plt.figure()
plt.pcolormesh(px,py,-1*np.ones_like(px),vmin=-1,vmax=3,rasterized=True)
plt.pcolormesh(px[::-1,::-1],py,np.log10(1.0*np.abs(dp_field.T)),vmin=-1,vmax=3,rasterized=True)
#plt.imshow(np.abs(dp_field.T),interpolation='bicubic')
draw_filleted(-0.008, 0.004, 0.0005, 0.01, -0.002)
draw_filleted(-0.010, 0.015, 0.001, 0.01, -0.0075)
draw_filleted(-0.0098, 0.012, 0.0005, 0.01, -0.006)

draw_filleted(0.009, 0.015, 0.001, -0.01, -0.0075)
cb=plt.colorbar()
#plt.set_clim([0,1e6])

plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.xlim([-0.01, 0.01])
plt.ylim([-0.01, 0.01])
cb.set_label("log$_{10}$($E_x$ [V/m])")

plt.gca().set_xticklabels([-10, -5, 0, 5, 10])
plt.gca().set_yticklabels([-10, -5, 0, 5, 10])

fig.set_size_inches(7.5,4.5)
plt.subplots_adjust(top=0.96, bottom=0.125)

plt.savefig("field_sim.pdf", dpi=300)

plt.show()


#massarr = np.logspace(-6, -1, 1e2)
massarr = np.logspace(-7,2,2e2)
ztouse = np.linspace( 0.000005, 0.000010, 1e2)

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

for i,mass in enumerate(massarr):

    lam = hbar*c/mass
    #print lam

    earr = np.zeros_like(ztouse)    
    for j, z in enumerate(ztouse):
        r, v = integrand(z, yy, evals, lam )
        earr[j] = np.trapz(v, r) * (1. - np.exp(-d/lam) )**2
    dEdy = np.gradient(earr, np.gradient(ztouse))
    dEdy_cent = np.median(dEdy)

    chiarr[i] = np.sqrt(sigF/(alpha*Edc*dEdy_cent*V))
    chiarr2[i] = np.sqrt(sigF/(np.abs(np.median(earr))*V*q))


#np.save("data/dphot_lim_%s_smallbead.npy"%str(space), np.vstack( (massarr, chiarr, chiarr2) ))

plt.figure()
plt.loglog( massarr, chiarr )
plt.loglog( massarr, chiarr2 )
plt.show()
