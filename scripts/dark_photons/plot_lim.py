import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.size'] = 15
#for d in [0.2, 0.5, 1, 2]: ##, 6, 8]:

d = 8
print d
cdat_small = np.load("data/dphot_lim_%s_smallbead.npy"%d)
cdat = np.load("data/dphot_lim_%s.npy"%d)

def plot_prev(n,col,fill,ec='',ls='--', below=False, alp=1.0, bf=False):
    d = np.loadtxt("dph_data/"+n+".txt",skiprows=1,delimiter=',')
    if(fill):
        if( not below):
            hh = plt.fill_between( d[:,0], d[:,1], np.ones_like(d[:,0]), color=col, edgecolor=ec, alpha=alp, linewidth=1.5)
        if(bf):
            hh.set_zorder(2)
        else:
            if( n == "sun"):
                solar_dat = np.loadtxt("dph_data/sun.txt",skiprows=1,delimiter=',')
                gpts = solar_dat[:,0] > 2.4e-5
                xx = np.hstack((2.4e-5,solar_dat[gpts,0]))
                yy = np.hstack((1.5e-7,solar_dat[gpts,1]))
                plt.fill_between(xx,1e-11*yy, yy, color=col, edgecolor=ec, alpha=0.05, linewidth=1.5)
            else:
                plt.fill_between( d[:,0], np.ones_like(d[:,0])*1e-11, d[:,1], color=col, edgecolor=ec, alpha=0.05, linewidth=1.5)
    else:
        print ls
        if(ls == '-.'):
            if( n == "dph_theory1" or n == "dph_theory2"):
                hh = plt.plot(d[:,0], d[:,1], linestyle=ls, color=col, linewidth=0.5)
                hh[0].set_zorder(1)
            else:
                plt.plot(d[:,0], d[:,1], linestyle=ls, color=col, linewidth=0.5)
        elif( ls != '-'):
            plt.plot(d[:,0], d[:,1], linestyle=ls, dashes=[2,2], color=col, linewidth=1.5)
        else:
            plt.plot(d[:,0], d[:,1], linestyle=ls, color=col, linewidth=1.)

fig=plt.figure()
prev_col = [0.5,0.75,1.0]
ec = [0,0,0.5]
astro_col = [0.92,0.92,0.92]
astro_ec = [0.5,0.5,0.5]

plot_prev("sun", astro_col, True, ec=astro_ec)
plot_prev("cmb", astro_col, True, ec=astro_ec)

plot_prev("nontherm_dm", [0,0.5,0], True, ec=[0,0.5,0], below=True)
plot_prev("topline", [0,0.8,0], True, ls='-', below=True) 
#plot_prev("dph_theory1", [0,0.8,0], True, ls='-', alp=0.05) 
plot_prev("dph_theory1", [0,0.6,0], False, ls='-.', alp=0.1) 
plot_prev("dph_theory2", [0,0.6,0], False, ls='-.', alp=0.1) 
plot_prev("topline", [0,0.6,0], False, ls='-.', alp=0.1) 
plot_prev("nontherm_dm", [0,0.4,0], False, ls='-', alp=0.1) 
plot_prev("sun", [0,0.8,0], True, below=True)

plot_prev("lsw", prev_col, True, ec=ec, bf=True) 
plot_prev("mwlsw", prev_col, True, ec=ec, bf=True) 
plot_prev("coulomb", prev_col, True, ec=ec, bf=True) 
plot_prev("rydberg", prev_col, True, ec=ec, bf=True) 

plot_prev("admx", [1,0.5,0], False) 
plot_prev("uwa", 'm', False) 
plot_prev("alps-ii", 'r', False) 

plt.text(1e-8, 1e-2, "Coulomb", color=ec, fontsize=14)
plt.text(1e0, 1e-2, "Rydberg", color=ec, fontsize=14)
plt.text(4e-3, 1e-4, "LSW", color=ec, fontsize=14)
plt.text(4e-10, 5e-7, "CMB", color=astro_ec, fontsize=14)
plt.text(8, 5e-7, "Sun", color=astro_ec, fontsize=14)
plt.text(5e-7, 5e-10, "ADMX", color=[1, 0.5, 0], fontsize=14)
plt.text(0.4, 3e-10, "ALPS-IIb", color='r', fontsize=14)
plt.text(7e-6, 1.3e-6, "UWA", color='m', fontsize=14)
plt.text(9e-5, 1.6e-7, "This proposal", color='k', fontsize=13, rotation=-20)
plt.text(6e-5, 7e-11, "Ultimate", color='k', fontsize=13, rotation=-18)

gpts = cdat_small[1,:] < 1e-3
plt.loglog(cdat_small[0,gpts], cdat_small[1,gpts],'k', linewidth=2.5)
frat = np.sqrt(1e-25/4e-21)
gpts = cdat[1,:]*frat < 1e-3
plt.loglog(cdat[0,gpts], cdat[1,gpts]*frat,'k--', dashes=[5,3], linewidth=2.5)
#plt.loglog(cdat[0,:], cdat[2,:],'g')
plt.xlim([1e-10, 1e3])
plt.ylim([1e-11, 1])

plt.gca().set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2])
#plt.gca().set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
for label in plt.gca().get_yticklabels()[1::2]:
    label.set_visible(False)

plt.xlabel("Dark photon mass, $m_{A'}$ [eV]")
plt.ylabel("Coupling, $\chi$")

fig.set_size_inches(6,4.5)
plt.subplots_adjust(bottom=0.14, right=0.99, left=0.15, top=0.95)
plt.savefig("dark_phot_sens.pdf")

plt.show()
