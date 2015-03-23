import numpy
from pylab import *
from scipy.interpolate import interp1d

d1,g1,r1,dep1,t1,e1,ee1,f1,ef1,s=numpy.loadtxt("PEC_combined_results.txt",unpack=True,skiprows=1)
f1=-f1*31.6e-15
inds=argsort(dep1)
d1=d1[inds]
f1=f1[inds]
r1=r1[inds]
dep1=dep1[inds]
g1=g1[inds]
t1=t1[inds]
s=s[inds]
inds=numpy.where(s == 0)
d1=d1[inds]
f1=f1[inds]
r1=r1[inds]
g1=g1[inds]
dep1=dep1[inds]
t1=t1[inds]

figure(figsize=(12,8))

inds = numpy.where(t1 == 300)
xd1=dep1[inds]
yf1=f1[inds]/numpy.min(f1[inds])
asp=r1[inds]
asps=numpy.unique(asp)
for i in range(0,len(asps)):
    gpts=numpy.where(asps[i] == asp)
    plot(xd1[gpts],yf1[gpts],'-o',label="PEC, temp="+str(300)+" r="+str(asps[i]))

#xscale('log')
#yscale('log')
xlabel('Aspect Ratio (D/H)')
ylabel('Force (N)')
xlim(1,5)
ylim(0.95,1.1)
title('Numerical Calculations, Aspect Ratio')
legend(loc="lower left",ncol=2)
savefig('depth_v_gridding.png')
#show()

clf()

inds = numpy.where(t1 == 300)
xd1=dep1[inds]
yf1=f1[inds]/numpy.min(f1[inds])
asp=r1[inds]
lens=numpy.unique(xd1)
for i in range(0,len(lens)):
    gpts=numpy.where(lens[i] == xd1)
    x=asp[gpts]
    y=yf1[gpts]/numpy.min(yf1[gpts])
    sinds=numpy.argsort(x)
    plot(x[sinds],y[sinds],'-o',label="T="+str(300)+" d="+str(lens[i]))

xlabel('Gridding Ratio')
ylabel('Force(Aspect)/min(Force)')
title('Aspect v Force Numerical Calculations')
legend(loc="lower left",ncol=4)
ylim(0.99,1.01)
savefig('depth_correction.png')
#show()
