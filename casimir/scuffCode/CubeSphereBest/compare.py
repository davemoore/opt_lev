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
savefig('analytic_v_numerical')
#show()

