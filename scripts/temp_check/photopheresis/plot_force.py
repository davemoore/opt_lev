import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

r = 7.5e-6 # radius, m
Rgas = 8.3 # J/(mol K)
eta = 1.8e-5 # Pa s
Mgas = 0.028 # kg/mol
dT = 0.005 #K
Tgas = 300 #K
gam = 1
kb = 1.38e-23 # J/K
msph = 3.2e-12 # kg
omega0 = 2*np.pi*150 # rad/s

def mean_free_path(vis, press, temp, Mgas):
    L1 = vis/press
    L2 = np.sqrt( np.pi*Rgas*Tgas/(2*Mgas) )
    return L1*L2

def visc( press ):
    Kn = mean_free_path( eta, press, Tgas, Mgas)/r
    ck = 0.31*Kn/(0.785 + 1.152*Kn + Kn**2)
    return eta*(0.619/(0.619+Kn))*(1 + ck)

def Gam0( press ):
    Kn = mean_free_path( eta, press, Tgas, Mgas)/r
    ck = 0.31*Kn/(0.785 + 1.152*Kn + Kn**2)
    return 6*np.pi*eta*r*(0.619/(0.619+Kn))*(1 + ck)/msph

pp = np.logspace(-2, 5, 1e3)
# plt.figure()
# plt.loglog( pp/1e2, visc(pp) )
# plt.loglog( pp/1e2, visc2(pp) )
# plt.show()
    
def Kf( p, gam ):
    #p = np.logspace(0,5,1e3) ## Pa

    A = 1.5*np.pi*Rgas*eta**2/Mgas * dT
    B = np.pi/6 * gam * r**2 * dT/Tgas

    p0 = np.sqrt( np.abs(A/B) )
    #print("p0 is: ", p0)
    #print( "fs p0: ", 3*eta*np.sqrt(Rgas*Tgas/(Mgas*gam))/r )
    return np.sqrt( np.abs(A*B) )/(p/p0 + p0/p)

print("force: ", Kf( 1e5, 0.6))


# plt.figure()
# plt.loglog( pp/1e2, Gam0(pp) )
# plt.show()

def P_at_fall(p, gam, E_0, alpha, beta, gamma):

    Fth = Kf(p,gam)
    G = (Gam0(p))
    alpha = np.abs(alpha)
    
    #P = (np.sqrt(np.abs(beta**2 + 4*E_0*(G+gamma)**2*msph*omega0**2 - 4*alpha*(G+gamma)*msph*omega0**2)) - beta)/(2*Fth*(G+gamma))
    P = (msph*omega0**2/(2*Fth**2)) * (np.sqrt( (Fth**2*gamma**2)/((alpha+G+beta+gamma)**2*msph**2*omega0**4) + 4*Fth**2/(msph*omega0**2)*(E_0 - G*kb*Tgas/(alpha+G+beta+gamma) - beta/(alpha+G+beta+gamma))) - Fth*gamma/((alpha + G+beta+gamma)*msph*omega0**2) )
    
    #P[  beta/(alpha+G) >= E_0 ] = 0

    for0 = G/(G+alpha+beta+gamma)*kb*Tgas
    for1 = beta/(G+alpha+beta+gamma) 
    for2 = gamma*Kf(p,gam)/((G+alpha+beta+gamma)*msph*omega0**2) #* P
    for3 = Kf(p,gam)**2/(msph*omega0**2) #* P**2
    
    # plt.figure()
    # plt.loglog(p, for0, label='f0')
    # plt.loglog(p, for1, label='f1')
    # plt.plot(p, for2, label='f2')
    # plt.plot(p, for3, label='f3')
    # plt.plot(p, for1+for2+for3, label='tot')
    # plt.plot(p, E_0*np.ones_like(p))
    # plt.legend()

    # plt.figure()
    # plt.loglog(p, for1, label='f1')
    # plt.plot(p, for2*P, label='f2')
    # plt.plot(p, for3*P**2, label='f3')
    # plt.plot(p, for1+for2*P+for3*P**2, label='tot')
    # plt.plot(p, E_0*np.ones_like(p))
    # plt.legend()
    
    # plt.figure()
    # plt.semilogx(p, P)
    # plt.show()
    
    return P

def Kf2( p, gam):
    Kn = mean_free_path( eta, p, Tgas, Mgas)/r
    return gam*((Kn + 0.129)/(Kn + 0.129*gam))*(Kn**2/(Kn**2 + 0.116))*((Kn**2 - 0.222*Kn + 0.131)/(Kn**2 + 0.015*Kn + 0.131))


## load Wenqiang's data
folder = "data"
file_list = glob.glob(folder+"/*.txt")

fig1=plt.figure()
x = []
y = []
for ii,i in enumerate(sorted(file_list)):
    print(i)
    a = np.loadtxt(i)
    b = np.transpose(a)
    x = np.append(x, b[0])
    if( ii < 3):
        y = np.append(y, b[1])
        yvals = b[1]
    else:
        y = np.append(y, b[1]/25*1.4)
        yvals = b[1]/25*1.4        

    #plt.semilogx( b[0], b[1]/25*1.4, 'o')
    plt.errorbar( b[0], yvals, xerr=b[0]*0.3, yerr=yvals*0.1, fmt='o' )
    
ind = np.argsort(x)

x = x[ind]
y = y[ind]

x = x*1e2 # it is in pascal


const = 1e-13

ffn_simp = lambda xx, ac, E_0: P_at_fall(xx, ac, E_0, 0, 0, 0)
ffn_simp1 = lambda xx, C, ac, E_0, beta: C*P_at_fall(xx, ac, E_0, 0, beta, 0)
ffn = lambda xx, C, ac, E_0, alpha, beta, gamma: C*P_at_fall(xx, ac, E_0, 0, beta, gamma)

spars = [ 4.73735391e+00,  3.87456239e-01,  5.56661460e-15, -5.26259165e+00,
          7.64852352e-14,  1.05517423e-08]
#errs = y[gpts]*0.01
#errs[errs == 0] = 1

gpts = x > 200
bp_simp, bcov_simp = opt.curve_fit( ffn_simp, x[gpts], y[gpts], p0=spars[1:3])
#bp_simp = spars[0:3]

print( "C,ac,E:", bp_simp, bp_simp[-1]/kb )

perr = np.sqrt(np.diag(bcov_simp))
print(perr)
print("Trap depth [K]: %.1e %.1e"%(bp_simp[-1]/kb, np.sqrt(bcov_simp[1,1])/kb))
gpts = x > 0
bp, bcov = opt.curve_fit( ffn, x[gpts], y[gpts], p0=spars )
#bp = spars
print(bp)

gpts = x > 100
bp_simp1, bcov_simp1 = opt.curve_fit( ffn_simp1, x[gpts], y[gpts], p0=[*spars[0:3], spars[4]])

# pp = np.logspace(0,5,1e3) ## Pa
# plt.figure()
# for1 = 0.1*bp[2]/Gam0(pp)
# for2 = 5e-10*(Kf(pp,bp[0]))**2/msph*omega0**2
# for3 = Kf(pp,bp[0])/Gam0(pp)
# plt.loglog(pp/1e2,for1)
# plt.loglog(pp/1e2,for2)
# plt.loglog(pp/1e2,for3)
# plt.loglog(pp/1e2,for1+for2+for3)
# plt.loglog(pp/1e2, np.sqrt((bp[1]-for1)/for2) )
# plt.plot(pp/1e2, np.ones_like(pp)*bp[1])

# plt.show()

#bp[1] = 1

#plt.figure()
#plt.loglog( p/1e2, A/p )
#plt.loglog( p/1e2, B*p )
#plt.loglog( p/1e2, K )

#plt.figure()

yy = ffn(pp, *bp)
plt.figure(fig1.number)
plt.semilogx( pp/1e2, ffn_simp(pp, *bp_simp), 'k--', label='Photophoresis only' )
plt.semilogx( pp/1e2, ffn_simp1(pp, *bp_simp1), 'k:', label='+laser noise' )
plt.semilogx( pp/1e2, yy, 'k', label='Full model' )
plt.ylim([0,65])
plt.xlim([1e-2, 1.1e3])
plt.xlabel("Pressure [mbar]")
plt.ylabel("CO2 laser power [units??]")
plt.legend(loc="upper left")
plt.savefig('co2_fit.pdf')

#plt.figure()
#plt.semilogx( pp/1e2, ffn(pp, *[5.68232572e-01, 1.04119581e-13, 1.3e-11]) )
#plt.ylim([0,65])
#plt.loglog( pp/1e2, Kf2(pp, bp[1]) )

plt.show()

