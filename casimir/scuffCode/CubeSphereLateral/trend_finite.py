import numpy
from pylab import *

L,W,grid,e,ee,f,ef,s=numpy.loadtxt("combined_results_temp.txt",unpack=True,skiprows=1)
f=-f*31.6e-15

lens=numpy.unique(L)
for i in range(0,len(lens)):
    gpts = numpy.where((L == lens[i]) & (grid == 0.5))
    x=W[gpts]
    y=f[gpts]
    inds=numpy.argsort(x)
    x=x[inds]
    y=y[inds]
    plot(x,y,label=str(lens[i]))
legend(loc='upper right')
yscale('log')
xlim(0,100)
savefig('lateral_force_finite')

lens=numpy.unique(L)
for i in range(0,len(lens)):
    gpts = numpy.where((L == lens[i]) & (grid == 0.5))
    x=W[gpts]
    y=f[gpts]
    inds=numpy.argsort(x)
    x=x[inds]
    y=y[inds]
    y=y/y[0]
    plot(x,y,label=str(lens[i]))
legend(loc='lower left')
yscale('log')
xlim(0,100)
ylim(1e-4,2)
savefig('lateral_force_drop_finite')
