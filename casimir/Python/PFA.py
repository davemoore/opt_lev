#!/usr/bin/env python

from scipy.constants import c
from scipy.integrate import quad,quadrature
import numpy
from numpy import *
from math import pi

#in these functions, side 1 is the gold cantilever and side 2 is the SiO2 sphere
#in addition, model refers to:
#   1 - Plasma
#   2 - Drude

def gamma(side):
    if(side == 1): #plane
        return 4.521E13
    elif(side == 2): #sphere
        return 
    else:
        print "Invalid Side: "+str(side)
        return 0

def omega_p(side):
    if(side == 1):
        return 1.37E16
    elif(side == 2):
        return 8.85938E15
    else:
        print "Invalid Side: "+str(side)
        return 0

def epsilon_i(omega, model, side):
    if (model == 1):
        return 1+pow((omega_p(side)/omega),2) #plasma model
    elif (model == 2):
        return 1+pow(omega_p(side),2)/(omega*(omega+gamma(side))) #drude model
    else:
        print "Invalid Model Index: "+str(model)
        return 0

def rp(omega,kappa,pol,model,side):
    epi = epsilon_i(omega, model, side)
    left=sqrt(pow(omega,2)*(epi-1)+pow(c,2)*kappa)
    if (pol == 1):
        right=c*kappa
    elif (pol ==2):
        right=c*kappa*epi
    else:
        print "Invalid Polarization Index: "+str(pol)
        return 0

    top = left-right
    bottom= left+right
    return top/bottom

def rsquared(omega, kappa, pol, model):
    return rp(omega, kappa, pol, model, 1)*rp(omega, kappa, pol, model, 2)

def etaE(model, L):

    def integrandK(kappa):
        def integrandO(omega):
            term1=numpy.log(1-rsquared(omega,kappa,1,model)*exp(-2*kappa*L))
            term2=numpy.log(1-rsquared(omega,kappa,2,model)*exp(-2*kappa*L))
            return (term1+term2)
        return kappa*quad(integrandO,0,kappa*c,limit=1000,epsrel=0.01)[0]
    
    ans = quad(integrandK,0,inf,limit=1000,epsrel=0.01)[0]
    return ans*-180*pow(L,3)/(c*pow(pi,4))
    

def PFA(model, L):
    return 0
