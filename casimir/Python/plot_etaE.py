from PFA import *
import numpy
from pylab import *

x = numpy.arange(-7,-4,0.1)
separation=pow(10,x)
etas = numpy.zeros(len(separation))
for i in range(0,len(separation)):
    etas[i] = etaE(1,separation[i])

plot(x,etas)
show()
