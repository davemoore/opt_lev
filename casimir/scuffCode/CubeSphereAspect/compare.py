import numpy
from pylab import *
from scipy.interpolate import interp1d

d1,g1,e1,ee1,f1,ef1,s=numpy.loadtxt("PEC_combined_results.txt",unpack=True,skiprows=1)
f1=-f1*31.6e-15
inds=argsort(d1)
d1=d1[inds]
f1=f1[inds]
g1=g1[inds]
s=s[inds]
inds=numpy.where(s == 0)
d1=d1[inds]
f1=f1[inds]
g1=g1[inds]

d2,g2,e2,ee2,f2,ef2,s2=numpy.loadtxt("combined_results.txt",unpack=True,skiprows=1)
f2=-f2*31.6e-15
inds=argsort(d2)
d2=d2[inds]
f2=f2[inds]
g2=g2[inds]
s2=s2[inds]
inds=numpy.where(s2 == 0)
d2=d2[inds]
f2=f2[inds]
g2=g2[inds]

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

print(f1)
print(f2)

datafile="../../Mathematica/calculated_vals.tsv"
PFA_datafile="../../Mathematica/calculated_pfa_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(PFA_datafile,unpack=True)
dist=dist*1e6

figure(figsize=(12,8))
gs=numpy.min(g1)
#for i in range(0,len(gs)):
inds = numpy.where(g1 == gs)
plot(d1[inds],f1[inds],'--',label="PEC, grid="+str(gs),color="black")
inds = numpy.where(g1 == 0.4)
plot(d1[inds],f1[inds],'-.',label="PEC, grid="+str(0.4),color="black")
gs=numpy.min(g2)
inds = numpy.where(g2 == gs)
plot(d2[inds],f2[inds],'--',label="FEC, grid="+str(gs),color="green")
plot(d4,f4,':',label="PEC, Large Cantilever",color="black")
plot(d3,f3,':',label="FEC, Large Cantilever",color="green")
plot(dist,fpfa,label="PFA",linestyle='-',color="black")
plot(dist,fright,label="SiO2/Au",linestyle='-',color="green")
plot(dist,ftemp,label="SiO2/Au T=300",linestyle='-',color="red")
xscale('log')
yscale('log')
xlabel('Distance (microns)')
ylabel('Force (N)')
title('Analytical (Dashed) v Numerical (Solid) Calculations')
legend(loc="lower left",ncol=2)
savefig('analytic_v_numerical')
#show()

#data points computed (through similar method) for correction due to aspect ratio L/R from PFA (Canaguier-Durand 2012)
cdx=[0,0.1,.2,0.4,0.6,0.8,1]
cdy=[1.0,.98,.95,.86,.78,.72,.68]

clf()
iPFA = interp1d(dist,fpfa)
gs=numpy.unique(g1)
for i in range(0,len(gs)):
    inds = numpy.where(g1 == gs[i])
    rPFA=f1[inds]/iPFA(d1[inds])
    plot(d1[inds]/2.5,rPFA,label="PFA, grid="+str(gs[i]))
plot(cdx,cdy,label="Canaguieier-Durand",linestyle=':',color="black")
#xscale('log')
xlim(0,3)
xlabel('Distance/Radius')
ylabel('(PFA/BEM) Force Ratio')
title('Comparion between Calculations, grid=1 micron')
legend()
#show()
savefig("pfa_v_pec.png")

clf()
inds=argsort(g1)
d1=d1[inds]
f1=f1[inds]
g1=g1[inds]
ds=numpy.unique(d1)
for i in range(0,len(ds)):
    inds=numpy.where(d1 == ds[i])
    plot(g1[inds],f1[inds]/f1[inds[0][0]],'--',label=str(ds[i]),alpha=.9)
plot([0.1,1.2],[1,1],linestyle=':',color='black')
ylim(0.2,1.1)
xlim(0.3,1)
xscale('log')
xlabel('Grid Scale Length')
ylabel('Force/Force(smallest gridding)')
title("Convergence in Grid Spacing")
legend(loc='lower left',title="Separation")
savefig("pfa_convergence.png")

clf()
inds=argsort(g1)
d1=d1[inds]
f1=f1[inds]
g1=g1[inds]
ds=numpy.unique(d1)
for i in range(0,len(ds)):
    inds=numpy.where(d1 == ds[i])
    plot(g1[inds],f1[inds]/f1[inds[0][0]],'--',label=str(ds[i]),alpha=.9)
plot([0.1,1.2],[1,1],linestyle=':',color='black')
ylim(0.8,1.1)
xlim(0.3,1)
xscale('log')
xlabel('Grid Scale Length')
ylabel('Force/Force(smallest gridding)')
title("Convergence in Grid Spacing")
legend(loc='lower left',title="Separation")
savefig("pfa_convergence_zoom.png")
