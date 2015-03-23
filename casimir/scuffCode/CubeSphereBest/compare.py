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

d1ts,g1ts,e1ts,ee1ts,f1ts,ef1ts,sts=numpy.loadtxt("../CubeSphere/PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1ts=-f1ts*31.6e-15
inds=argsort(d1ts)
d1ts=d1ts[inds]
f1ts=f1ts[inds]
g1ts=g1ts[inds]
sts=sts[inds]
inds=numpy.where(sts == 0)
d1ts=d1ts[inds]
f1ts=f1ts[inds]
g1ts=g1ts[inds]

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

d2ts,g2ts,e2ts,ee2ts,f2ts,ef2ts,sts2=numpy.loadtxt("../CubeSphere/combined_results_temp.txt",unpack=True,skiprows=1)
f2ts=-f2ts*31.6e-15
inds=argsort(d2ts)
d2ts=d2ts[inds]
f2ts=f2ts[inds]
g2ts=g2ts[inds]
sts2=sts2[inds]
inds=numpy.where(sts2 == 0)
d2ts=d2ts[inds]
f2ts=f2ts[inds]
g2ts=g2ts[inds]

datafile="../../Mathematica/calculated_vals.tsv"
PFA_datafile="../../Mathematica/calculated_pfa_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(PFA_datafile,unpack=True)
dist=dist*1e6

figure(figsize=(12,8))

gst=numpy.min(g1t)
inds = numpy.where(g1t == gst)
plot(d1t[inds],f1t[inds],'-o',label="PEC 300K, grid="+str(gst),color="black")
inds = numpy.where(g1t == 0.5)
plot(d1t[inds],f1t[inds],'-.',label="PEC 300K, grid="+str(0.4),color="black")

gst=numpy.min(g1ts)
inds = numpy.where(g1ts == gst)
plot(d1ts[inds],f1ts[inds],'--',label="PEC 300K, small geometry, grid="+str(gst),color="black")

gs=numpy.min(g2t)
inds = numpy.where(g2t == gs)
plot(d2t[inds],f2t[inds],'-o',label="FEC 300K, grid="+str(gs),color="green")
inds = numpy.where(g2t == 0.5)
plot(d2t[inds],f2t[inds],'-.',label="FEC 300K, grid="+str(0.5),color="green")

gs=numpy.min(g2ts)
inds = numpy.where(g2ts == gs)
plot(d2ts[inds],f2ts[inds],'--',label="FEC 300K, small geometry, grid="+str(gs),color="green")

plot(dist,fpfa,label="PFA",linestyle=':',color="black")
plot(dist,fright,label="SiO2/Au",linestyle=':',color="green")
plot(dist,ftemp,label="SiO2/Au T=300",linestyle=':',color="red")
xlim(1,30)
#xscale('log')
yscale('log')
ylim(1e-22,1e-14)
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
ylim(0.9,1.05)
xlim(0.3,0.5)
xlabel('Grid Scale Length')
ylabel('Force/Force(smallest gridding)')
title("Convergence in Grid Spacing")
legend(loc='lower left',title="Separation",ncol=2)
savefig("pfa_convergence_zoom_best.png")

clf()
id1ts = interp1d(d1ts,f1ts)
gpts=where(g2ts == numpy.min(g2ts))
scatter(d2ts[gpts],f2ts[gpts]/id1ts(d2ts[gpts]))
savefig('conductivity_correction.png')
