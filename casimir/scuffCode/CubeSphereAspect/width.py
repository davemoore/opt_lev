import numpy
from pylab import *
from scipy.interpolate import interp1d

d1,g1,a1,t1,e1,ee1,f1,ef1,s=numpy.loadtxt("PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1=-f1*31.6e-15
inds=argsort(d1)
d1=d1[inds]
f1=f1[inds]
a1=a1[inds]
g1=g1[inds]
s=s[inds]
inds=numpy.where(s == 0)
d1=d1[inds]
f1=f1[inds]
a1=a1[inds]
g1=g1[inds]

d2,g2,a2,t2,e2,ee2,f2,ef2,s2=numpy.loadtxt("combined_results_temp.txt",unpack=True,skiprows=1)
f2=-f2*31.6e-15
inds=argsort(d2)
d2=d2[inds]
f2=f2[inds]
a2=a2[inds]
g2=g2[inds]
s2=s2[inds]
inds=numpy.where(s2 == 0)
d2=d2[inds]
f2=f2[inds]
a2=a2[inds]
g2=g2[inds]

figure(figsize=(12,8))
gs=numpy.unique(g1)

for j in range(0,len(gs)):
    inds = numpy.where(g1 == gs[j])
    xd1=d1[inds]
    yf1=f1[inds]
    asp=a1[inds]
    asps=numpy.unique(asp)
    for i in range(0,len(asps)):
        gpts=numpy.where(asps[i] == asp)
        plot(xd1[gpts],yf1[gpts],'-o',label="PEC, grid="+str(gs[j])+" asp="+str(asps[i]))

#gs=numpy.min(g2)
#inds = numpy.where(g2 == gs)
#plot(d2[inds],f2[inds],'--',label="FEC, grid="+str(gs),color="green")
xscale('log')
yscale('log')
xlabel('Distance (microns)')
ylabel('Force (N)')
xlim(10,30)
title('Numerical Calculations, Aspect Ratio')
legend(loc="lower left",ncol=2)
savefig('analytic_v_numerical')
#show()

clf()
gs=numpy.unique(g1)

for j in range(0,len(gs)):
    inds = numpy.where(g1 == gs[j])
    xd1=d1[inds]
    yf1=f1[inds]
    asp=a1[inds]
    lens=numpy.unique(xd1)
    for i in range(0,len(lens)):
        gpts=numpy.where(lens[i] == xd1)
        plot(asp[gpts],yf1[gpts]/numpy.min(yf1[gpts]),'-o',label="g="+str(gs[j])+" l="+str(lens[i]))

xlabel('Aspect')
ylabel('Force (N)')
title('Aspect v Force Numerical Calculations')
legend(loc="lower left",ncol=3)
ylim(0.75,1.5)
savefig('aspect_correction.png')
#show()
