import numpy
from pylab import *
from scipy.interpolate import interp1d

d1t,g1t,e1t,ee1t,f1t,ef1t,st=numpy.loadtxt("PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1t=-f1t*31.6e-15*1e18
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
f2t=-f2t*31.6e-15*1e18
inds=argsort(d2t)
d2t=d2t[inds]
f2t=f2t[inds]
g2t=g2t[inds]
s2t=s2t[inds]
inds=numpy.where(s2t == 0)
d2t=d2t[inds]
f2t=f2t[inds]
g2t=g2t[inds]

datafile="../../Mathematica/calculated_vals.tsv"
PFA_datafile="../../Mathematica/calculated_pfa_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(PFA_datafile,unpack=True)
dist=dist*1e6
ftemp=ftemp*1e18

figure(figsize=(12,8))
xnew=np.arange(1, 30, 0.1)

gst=numpy.min(g1t)
inds = numpy.where(g1t == gst)
s = interp1d(d1t[inds],log(f1t[inds]),kind='cubic')
plot(xnew,exp(s(xnew)),'--',label="PEC Limit, g="+str(gst),color="black")

gs=numpy.min(g2t)
gs=0.4
inds = numpy.where(g2t == gs)
xnew=np.arange(numpy.min(d2t[inds]), 30, 0.1)
s = interp1d(d2t[inds],log(f2t[inds]),kind='cubic')
plot(xnew,exp(s(xnew)),'-',label="Prediction, g="+str(gs),color="black")
plot(xnew,exp(s(xnew))*1.08,':',label="Prediction Uncertainty, g="+str(gs),color="black")
plot(xnew,exp(s(xnew))*0.92,':',color="black")
print(d2t[inds])
print(f2t[inds])

plot([1,30],[1e3,1e3],':',color="black")
plot([1,30],[1e2,1e2],':',color="black")
plot([1,30],[1e1,1e1],':',color="black")
plot([1,30],[1,1],':',color="black")
plot([1,30],[1e-1,1e-1],':',color="black")
plot([1,30],[1e-2,1e-2],':',color="black")
plot([1,30],[1e-3,1e-3],':',color="black")

plot(dist,ftemp,label="SiO2/Au PFA",linestyle='--',color="green")
xlim(2.5,30)
#xscale('log')
yscale('log')
ylim(1e-2,5e2)
xlabel('Distance (microns)')
ylabel('Force (1e-18 N)')
title('Numerical Prediction for Casimir Force')
legend(loc="lower left")
savefig('prediction')
#show()
