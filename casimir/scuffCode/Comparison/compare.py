import numpy
from pylab import *
from scipy.interpolate import interp1d

d1,e1,ee1,f1,ef1=numpy.loadtxt("full.txt",unpack=True)
f1=-f1*31.6e-15
inds=argsort(d1)
d1=d1[inds]
f1=f1[inds]

d2,e2,ee2,f2,ef2=numpy.loadtxt("PEC.txt",unpack=True)
f2=-f2*31.6e-15
inds=argsort(d2)
d2=d2[inds]
f2=f2[inds]

d3,e3,ee3,f3,ef3=numpy.loadtxt("temp.txt",unpack=True)
f3=-f3*31.6e-15
inds=argsort(d3)
d3=d3[inds]
f3=f3[inds]

datafile="../../Mathematica/calculated_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(datafile,unpack=True)
dist=dist*1e6

plot(d2,f2,label="PEC")
plot(d1,f1,label="SiO2/Au")
plot(d3,f3,label="SiO2/Au T=300")

plot(dist,fpfa,label="PFA",linestyle='dashed')
plot(dist,fright,label="SiO2/Au",linestyle='dashed')
plot(dist,ftemp,label="SiO2/Au T=300",linestyle='dashed')
xscale('log')
yscale('log')
legend()
savefig('analytic_v_numerical')

clf()
iPFA = interp1d(dist,fpfa)
rPFA=iPFA(d2)/f2
plot(d2,rPFA)
xscale('log')
yscale('log')
savefig("pfa_v_pec.png")
