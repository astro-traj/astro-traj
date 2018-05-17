import numpy as np
from scipy.optimize import brentq


class relation():
    def __init__(self,version = 'old'):

        if version == 'old':
            
            self.M10 = 11.590
            self.M11 = 1.195
            self.N10 = 0.0351
            self.N11 = -0.0247
            self.B10 = 1.376
            self.B11 = -0.826
            self.G10 = 0.608
            self.G11 = 0.329

            def G(z): return self.G10 + (self.G11*z/(z+1))
            self.G = G

        if version == 'new':
            
            self.M10 = 11.339
            self.M11 = 0.692
            self.N10 = 0.005
            self.N11 = 0.689
            self.B10 = 3.344
            self.B11 = -2.079
            
            def G(z): return 0.966
            self.G = G

        def M1(z): return 10.**(self.M10 + (self.M11*z/(z+1)))
        def N(z): return self.N10 + (self.N11*z/(z+1))
        def B(z): return self.B10 + (self.B11*z/(z+1))

        self.M1 = M1
        self.N = N
        self.B = B


        

    def getMhalo(self,mstar,z):
        def BCE(exp_m):
            mhalo=10.**exp_m
            Mtot = mhalo + mstar
            term1 = mstar/Mtot
            term2 = 2.*self.N(z)
            term3 = ((Mtot/self.M1(z))**-self.B(z)) + ((Mtot/self.M1(z))**self.G(z))

            return (term2/term3)-term1
        
            
        mexp = brentq(BCE,1,40)
        return 10.**mexp
        
        
        
