import numpy
from pylab import *
from scipy.interpolate import interp1d

d1t,g1t,e1t,ee1t,f1t,ef1t,st=numpy.loadtxt("PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1t=-f1t*31.6e-15
inds=argsort(d1t)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]
st=st[inds]
inds=numpy.where(st == 0)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]

d2t,g2t,e2t,ee2t,f2t,ef2t,s2t=numpy.loadtxt("combined_results_temp.txt",unpack=True,skiprows=1)
f2t=-f2t*31.6e-15
inds=argsort(d2t)
d2t=d2t[inds]
f2t=f2t[inds]
g2t=g2t[inds]
s2t=s2t[inds]
inds=numpy.where(s2t == 0)
d2t=d2t[inds]
f2t=f2t[inds]
g2t=g2t[inds]

d3,e3,ee3,f3,ef3=numpy.loadtxt("../Comparison/full.txt",unpack=True)
f3=-f3*31.6e-15
inds=argsort(d3)
d3=d3[inds]
f3=f3[inds]

d4,e4,ee4,f4,ef4=numpy.loadtxt("../Comparison/PEC.txt",unpack=True)
f4=-f4*31.6e-15
inds=argsort(d4)
d4=d4[inds]
f4=f4[inds]

datafile="../../Mathematica/calculated_vals.tsv"
PFA_datafile="../../Mathematica/calculated_pfa_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(PFA_datafile,unpack=True)
dist=dist*1e6

figure(figsize=(12,8))

gst=numpy.min(g1t)
inds = numpy.where(g1t == gst)
plot(d1t[inds],f1t[inds],'-.',label="PEC 300K, grid="+str(gst),color="black")
inds = numpy.where(g1t == 0.4)
#plot(d1t[inds],f1t[inds],'-.',label="PEC 300K, grid="+str(0.4),color="orange")

gs=numpy.min(g2t)
inds = numpy.where(g2t == gs)
plot(d2t[inds],f2t[inds],'-.',label="FEC 300K, grid="+str(gs),color="green")

plot(d4,f4,':',label="PEC, Large Cantilever",color="black")
plot(d3,f3,':',label="FEC, Large Cantilever",color="green")
plot(dist,fpfa,label="PFA",linestyle='-',color="black")
plot(dist,fright,label="SiO2/Au",linestyle='-',color="green")
plot(dist,ftemp,label="SiO2/Au T=300",linestyle='-',color="red")
xlim(0.1,30)
xscale('log')
yscale('log')
xlabel('Distance (microns)')
ylabel('Force (N)')
title('Analytical (Dashed) v Numerical (Solid) Calculations')
legend(loc="lower left",ncol=2)
savefig('analytic_v_numerical_best')
#show()

#data points computed (through similar method) for correction due to aspect ratio L/R from PFA (Canaguier-Durand 2012)
cdx=[0,0.1,.2,0.4,0.6,0.8,1]
cdy=[1.0,.98,.95,.86,.78,.72,.68]

clf()
iPFA = interp1d(dist,fpfa)
gs=numpy.unique(g1t)
for i in range(0,len(gs)):
    inds = numpy.where(g1t == gs[i])
    rPFA=f1t[inds]/iPFA(d1t[inds])
    plot(d1t[inds]/2.5,rPFA,label="PFA, grid="+str(gs[i]))
plot(cdx,cdy,label="Canaguieier-Durand",linestyle=':',color="black")
#xscale('log')
xlim(0,3)
xlabel('Distance/Radius')
ylabel('(PFA/BEM) Force Ratio')
title('Comparion between Calculations, grid=1 micron')
legend()
#show()
savefig("pfa_v_pec_best.png")

clf()
inds=argsort(g1t)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]
ds=numpy.unique(d1t)
for i in range(0,len(ds)):
    inds=numpy.where(d1t == ds[i])
    plot(g1t[inds],f1t[inds]/f1t[inds[0][0]],'--',label=str(ds[i]),alpha=.9)
plot([0.1,1.2],[1,1],linestyle=':',color='black')
ylim(0.2,1.1)
xlim(0.3,1)
xscale('log')
xlabel('Grid Scale Length')
ylabel('Force/Force(smallest gridding)')
title("Convergence in Grid Spacing")
legend(loc='lower left',title="Separation")
savefig("pfa_convergence_best.png")

clf()
inds=argsort(g1t)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]
ds=numpy.unique(d1t)
for i in range(0,len(ds)):
    inds=numpy.where(d1t == ds[i])
    plot(g1t[inds],f1t[inds]/f1t[inds[0][0]],'--',label=str(ds[i]),alpha=.9)
plot([0.1,1.2],[1,1],linestyle=':',color='black')
ylim(0.8,1.1)
xlim(0.3,1)
xscale('log')
xlabel('Grid Scale Length')
ylabel('Force/Force(smallest gridding)')
title("Convergence in Grid Spacing")
legend(loc='lower left',title="Separation")
savefig("pfa_convergence_zoom_best.png")
