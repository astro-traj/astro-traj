# -*- coding: utf-8 -*-
# Copyright (C) Chase Kimball, Michael Zevin (2018)
#
# This file is part of the progenitor package.
#
# progenitor is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# progenitor is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with progenitor.  If not, see <http://www.gnu.org/licenses/>.

__author__ = ['Michael Zevin <michael.zevin@ligo.org>', 'Chase Kimball <charles.kimball@ligo.org']
__credits__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['Relation']


import numpy as np
from scipy.optimize import brentq


class Relation():
    """
    Utilizes Stellar Mass -- Halo Mass relation for inferring dark matter halo masses
    """

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
        """
        Gets dark matter halo mass
        """
        def BCE(exp_m):
            mhalo=10.**exp_m
            Mtot = mhalo + mstar
            term1 = mstar/Mtot
            term2 = 2.*self.N(z)
            term3 = ((Mtot/self.M1(z))**-self.B(z)) + ((Mtot/self.M1(z))**self.G(z))

            return (term2/term3)-term1
        
            
        mexp = brentq(BCE,1,40)
        return 10.**mexp
        
