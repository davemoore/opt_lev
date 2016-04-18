import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu

atom_dat = np.loadtxt( "atom_inter_lims.txt", delimiter=",", skiprows=1 )

our_lims = np.load("final_cham_limits_paper.npy")


fig=plt.figure()
plt.fill_between( atom_dat[:,0], atom_dat[:,1],1e2*np.ones_like(atom_dat[:,1]), edgecolor='none', color=[0.7,0.9,1] )
#xvals = np.hstack((1./our_lims[::-1,1],1./our_lims[:,2]))
#yvals = np.hstack((our_lims[::-1,0],our_lims[:,0]))
xvals = np.hstack([1e-12,1./our_lims[:,1]])
yvals = np.hstack([our_lims[0,0],our_lims[:,0]])
plt.fill_between( xvals, yvals, np.ones_like(yvals)*100, edgecolor='k', linewidth=2, color = [0.6, 0.6, 0.6])
#plt.plot( 1./our_lims[:,1], our_lims[:,0], 'k' )
#plt.plot( 1./our_lims[:,2], our_lims[:,0], 'k' )
plt.text(1e-6, 30, "These results", color=[1,1,1], fontsize=14, ha="center", va="center")
plt.text(1.4e-7, 0.75, "Atom\n interferometry", color=[0,0,0], fontsize=11, ha="center", va="center")
plt.text(1.4e-8, 3.5, "Neutron interferometry", color=[0,0,0], fontsize=11, ha="left", va="center")
plt.text(0.015, 16, r"E$\"\mathrm{o}$t-Wash", color=[0,0,0], fontsize=11, ha="left", va="center")
plt.text(8e-5, 2.6, r"$\Lambda$=2.4 meV", color=[0,0,0], fontsize=13, ha="left")


def make_line_seg( point, sty="-", left = True  ):
    m = np.log10( 2.5e-4/5e-5 )/10.
    deltax = 0.3*point
    upper = [point+deltax, 3.0]
    lower = [point-deltax, 1.9]
    #plt.plot( [lower[0]*0.85, upper[0]*1.15], [lower[1]*0.9,upper[1]*1.1], 'k'+sty, linewidth=1.5 )
    ax=plt.gca()
    if( left ):
        #ax.arrow( point, 2.78, -0.8*point, 0, fc='k', ec='k', linewidth=1.5, head_length=0.04*point, head_width=0.4 )
        plt.plot( [lower[0],upper[0]], [lower[1],upper[1]], 'k', linewidth=1.5 )
        plt.fill_between([1e-8, lower[0], upper[0]],[lower[1],lower[1],upper[1]],[upper[1],upper[1],upper[1]],edgecolor='k',facecolor='none',color='k',hatch=" \\\\\\\\ ", linewidth=0)

    else:
        #ax.arrow( point, 2.78, 4*point, 0, fc='k', ec='k', linewidth=1.5, head_length=1.2*point, head_width=0.4 )
        deltax = 0
        upper = [point+deltax, 3.0]
        lower = [point-deltax, 1.9]
        plt.plot( [lower[0],upper[0]], [lower[1],upper[1]], 'k', linewidth=1.5 )
        plt.fill_between([lower[0],1],[lower[1],lower[1]],[upper[1],upper[1]],edgecolor='k',facecolor='none',color='k',hatch='////', linewidth=0)

## neutron limits:  1.9e-7 at 2.4 meV
make_line_seg( 1.9e-7)

## eot-wash 1/50.3 at 2.4 meV
#make_line_seg( 1/50.3, left=False)
## plot's amol's limits
amol_lims = np.loadtxt("amol_lims.txt",delimiter=",", skiprows=1)
x = 1./amol_lims[:-2,0]
y = 2.4*(amol_lims[:-2,1])**0.2
plt.plot(x,y,'k',linewidth=1.5)
plt.fill_between(x,y,np.ones_like(y)*y[-1],edgecolor='k',facecolor='none',color='k',hatch='////', linewidth=0)

plt.plot([1e-12,1],[2.4,2.4],'k', linewidth=1.5)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlim([1e-8,1]) 
plt.ylim([0.1,100]) 

plt.xlabel(r"1/$\beta$ = $M/M_{Pl}$")
plt.ylabel("$\Lambda$ [meV]")

fig.set_size_inches(5,3.5)
plt.subplots_adjust(top=0.96,right=0.97,left=0.14,bottom=0.15)
plt.savefig("plots/limit_plot.pdf")


## now projected limits
if(True):
    tri_point = [2.17e-5, 4.6]
    lam_min = 2.3
    p2log = [1./bu.beta_max(lam_min),lam_min]
    ll=bu.beta_max(lam_min)/tri_point[0]
    lam_min = 0.37
    tri_point2=[ tri_point[0]*(4.6/lam_min)**1.7, lam_min]
    min_force_sens = 2e-17/np.sqrt(1000.)
    plt.loglog( [tri_point[0]*1e-2, tri_point2[0]], [tri_point2[1], tri_point2[1]], 'k:', linewidth=2 )
    p1a = [1./our_lims[0,1],1./our_lims[-1,1]]
    p2a = np.array([our_lims[0,0],our_lims[-1,0]*0.85])*min_force_sens/1e-16
    p1 =[p1a[0], p2a[0]]
    p2 =[p1a[1], p2a[1]]
    n = np.log10(p2[1]/p1[1])/np.log10(p2[0]/p1[0])
    p3 = [p2[0]*(lam_min/p2[1])**(1./n),lam_min]
    lam_min = 100.
    p4 = [p2[0]*(lam_min/p2[1])**(1./n), lam_min*7.0]
    plt.loglog( [p3[0],p4[0]], [p3[1],p4[1]], 'k:', linewidth=2 )

    plt.savefig("plots/limit_plot_proj.pdf")

plt.show()
