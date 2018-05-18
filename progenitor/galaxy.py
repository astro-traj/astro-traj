# -*- coding: utf-8 -*-
# Copyright (C) Michael Zevin (2018)
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
__all__ = ['Galaxy_Models', 'Hernquist_NFW', 'Belczynski_2002', 'Miyamoto_Nagai_NFW']

"""`Galaxy_Models` class which includes children `Miyamaoto_Nagai_Hernquist` and `NFW_Hernquist`
"""

import numpy as np

import astropy.units as units
import astropy.constants as constants


class Galaxy_Models(object):

    def __init__(self, Mspiral, Mbulge, Mhalo, R_eff, distance, redshift, h, rcut=None):
        """
        Galaxy class. Masses in Msun, distances in kpc. Immediately converted to SI.
        """
        self.Mspiral = Mspiral*units.Msun.to(units.kg)
        self.Mbulge = Mbulge*units.Msun.to(units.kg)
        self.Mhalo = Mhalo*units.Msun.to(units.kg)
        self.R_eff = R_eff*units.kpc.to(units.m)
        self.distance = distance*units.Mpc.to(units.m)
        self.redshift = redshift
        self.h = h
        # distance at which we set the potential to drop to 0, if not specified set to R_200
        if rcut:
            self.rcut = rcut*units.kpc.to(units.m)
        self.G = constants.G.value


class Hernquist_NFW(Galaxy_Models):
    """
    CURRENTLY PREFERRED POTENTIAL
    Galaxy with Hernquist potential for stellar component and dark matter potential and NFW profile for dark matter

    Paper References:
        Hernquist 1990: http://adsabs.harvard.edu/abs/1990ApJ...356..359H
             Ubulge is as in equation 5 in that paper
        Navarro, Frenk, White 1996: http://adsabs.harvard.edu/abs/1996ApJ...462..563N
            Uhalo corresponds to density profile in equation 3 in that paper.
            For explicit equation for this  potential see, say, Naray et al 2009 Equation 1 (http://iopscience.iop.org/article/10.1088/0004-637X/692/2/1321/meta)


    """
    def __init__(self, Mspiral, Mbulge, Mhalo, R_eff, distance, redshift, h, rcut=None):
        # Shared galaxy params
        Galaxy_Models.__init__(self, Mspiral, Mbulge, Mhalo, R_eff, distance, redshift, h, rcut)
        # define parameters that are used in this model
        # relationship between R_effective and abulge from Hernquist 1990 (eq. 38)
        self.abulge = self.R_eff / 1.8153

        # critical density of the universe (SI units)
        self.rho_crit = 1.879 * self.h**2 * 10**(-26)

        # empirical formula for c_200 from Duffy et al. 2008, using a redshift of 0
        def c_200(self, M_200):
            if self.redshift < 1:
                return 10**(0.76 - 0.1*np.log10(M_200 / ((2e12/self.h) * constants.M_sun.value)))
            elif (self.redshift >= 1) & (self.redshift < 2):
                return 10**(0.63 - 0.08*np.log10(M_200 / ((2e12/self.h) * constants.M_sun.value)))
            elif self.redshift >= 2:
                return 10**(0.53 - 0.03*np.log10(M_200 / ((2e12/self.h) * constants.M_sun.value)))
        # approximate M_200 = M_halo
        self.c = c_200(self, self.Mhalo)
        # R_200 is defined where density falls to 200 times the critical density
        self.R_200 = (3*self.Mhalo / (800*np.pi*self.rho_crit))**(1./3)
        # definition of scale radius
        self.Rs = self.R_200 / self.c
        # from integrating density distribution to R_200
        self.rho0 = self.Mhalo / (4*np.pi*self.Rs**3 * (np.log(1+self.c) - self.c/(1+self.c)))

        # if rcut not specified, set rcut to R_200
        if not rcut:
            self.rcut = self.R_200

    def Uhalo(self, r):
        """
        Halo contribution to potential divided by system mass
        """
        return -4*np.pi*self.G*self.rho0*(self.Rs**3)*np.log(1+(r/self.Rs))/r

    def Ubulge(self, r):
        """
        Bulge contribution to potential divided by system mass
        """
        return -self.G*self.Mbulge/(r+self.abulge)

    def Vdot(self, time, COORD):
        """
        Vdot calculation using gradients of Uhalo and Ubulge
        """
        x,y,z = COORD[3:]
        xdot,ydot,zdot = COORD[:3]
        r = np.sqrt((x**2)+(y**2)+(z**2))

        gradUx = self.halo(x,r)+self.bulge(x,r)
        gradUy = self.halo(y,r)+self.bulge(y,r)
        gradUz = self.halo(z,r)+self.bulge(z,r)

        return np.array([-gradUx,-gradUy,-gradUz, xdot, ydot, zdot])

    def halo(self, r_i, r):
        """
        gradient of Uhalo
        """
        term1 = r_i*np.log(1+(r/self.Rs))/(r**3)
        term2 = -r_i/((r**2)*(self.Rs+r))
        return self.G*4.0*np.pi*self.rho0*(self.Rs**3)*(term1+term2)


    def bulge(self, r_i, r):
        """
        gradient of Ubulge
        """
        return self.G*self.Mbulge*r_i/(r*((self.abulge+r)**2))



class Belczynski_2002(Galaxy_Models):
    """
    WARNING: NOT AT PRODUCTION LEVEL

    Galaxy with Hernquist potential for stellar component and dark matter potential from Belczynski 2002

    Paper References:
         Hernquist 1990: http://adsabs.harvard.edu/abs/1990ApJ...356..359H
             Ubulge is as in equation 5 in that paper
         Belzcynski et al 2002: http://iopscience.iop.org/article/10.1086/339860/meta
             Uhalo is as in equation 6 in that paper with cutoff at rcut such that
             Uhalo goes like 1/r for r>rcut



    """
    def __init__(self, Mspiral, Mbulge, Mhalo, R_eff, h, rcut):
        # Shared galaxy params
        Galaxy_Models.__init__(self, Mspiral, Mbulge, Mhalo, R_eff, h, rcut)
        # define parameters that are used in this model
        # Used in Belczynski+2002 halo potential, should be where DM density profile flattens FIXME
        self.rcore = 100.0*constants.kpc.value
        # Use relationship between R_effective and abulge from Hernquist 1990 (eq. 38)
        self.abulge = self.R_eff / 1.8153
        #FIXME: Need to make sure U(r>rcut) is implemented properly. I think it's right, but I want someone else's eyes on it
    def Uhalo(self, r):
        """
        Halo contribution to potential divided by system mass
        """
        if r>self.rcut:
            return self.rcut*Uhalo(self.rcut)/r
        else:
            term1 = .5*np.log(1+((r/self.rcore)**2))
            term2 = rcore*np.arctan(r/self.rcore)/r
            return -1.0*self.G*self.Mhalo*(term1+term2)/self.rcore

    def Ubulge(self, r):
        """
        Bulge contribution to potential divided by system mass
        """
        return -self.G*self.Mbulge/(r+abulge)

    def Vdot(self, time, COORD):
        """
        Vdot calculation using gradients of Uhalo and Ubulge
        """
        x,y,z = COORD[3:]
        xdot,ydot,zdot = COORD[:3]
        r = np.sqrt((x**2)+(y**2)+(z**2))

        gradUx = self.halo(x,r) + self.bulge(x,r)
        gradUy = self.halo(y,r) + self.bulge(y,r)
        gradUz = self.halo(z,r) + self.bulge(z,r)

        return np.array([-gradUx,-gradUy,-gradUz, xdot, ydot, zdot])

    def halo(self, r_i, r):
        """
        gradient of Uhalo
        """
        term1 = r_i/((rcore**2)+(r**2))
        term2 = r_i/((r**2)*(1.0+((r/rcore)**2)))
        term3 = -1.0*r_i*rcore*np.arctan(r/rcore)/(r**3)

        if r>self.rcut:
            P = self.rcut*Uhalo(self.rcut)
            return -P*r_i/(r**1.5)

        else: return -self.G*self.Mhalo*(term1+term2+term3)/rcore


    def bulge(self, r_i, r):
        """
        gradient of Ubulge
        """
        return self.G*self.Mbulge*r_i/(r*((self.abulge+r)**2))


class Miyamoto_Nagai_NFW(Galaxy_Models):
    """
    WARNING: NOT AT PRODUCTION LEVEL

    Spiral Galaxy with disk and bulge potential from Miyamoto & Nagai 1975 and NFW profile for dark matter
    NOTE: This potential model is not working yet. We only have 1 measurement for the stellar mass component, not a separate bulge and disk mass
    Paper References:
        Miyamoto and Nagai 1975: http://adsabs.harvard.edu/abs/1975PASJ...27..533M
            Udisk, Ubulge are as in Equation 4 in that paper

        Navarro, Frenk, White 1996: http://adsabs.harvard.edu/abs/1996ApJ...462..563N
            Uhalo corresponds to density profile in equation 3 in that paper.
            For explicit equation for this  potential see, say, Naray et al 2009 Equation 1 (http://iopscience.iop.org/article/10.1088/0004-637X/692/2/1321/meta)


    """
    def __init__(self, Mspiral, Mbulge, Mhalo, R_eff, h, rcut):
        # Shared galaxy params
        Galaxy_Models.__init__(self, Mspiral, Mbulge, Mhalo, R_eff, h, rcut)
        # define parameters that are used in this model
        # relationship between R_effective and abulge from Hernquist 1990 (eq. 38)
        self.abulge = self.R_eff / 1.8153
        # critical density of the universe (SI units)
        self.rho_crit = 1.879 * self.h**2 * 10**(-26)

        # empirical formula for c_200 from Duffy et al. 2008, using a redshift of 0
        def c_200(M_200):
            return 10**(0.76 - 0.1*np.log10(M_200 / (2.78*10**12 * constants.M_sun.value)))
        # approximate M_200 = M_halo
        self.c = c_200(self.Mhalo)
        # R_200 is defined where density falls to 200 times the critical density
        self.R_200 = (3*self.Mhalo / (800*np.pi*self.rho_crit))**(1./3)
        # definition of scale radius
        self.Rs = self.R_200 / self.c
        # from integrating density distribution to R_200
        self.rho0 = self.Mhalo / (4*np.pi*self.Rs**3 * (np.log(1+self.c) - self.c/(1+self.c)))

    def Udisk(self, r, z):
        """
        Disk contribution to potential divided by system mass
        """

        return
    def Uhalo(self, r):
        """
        Halo contribution to potential divided by system mass
        """
        return -4*np.pi*self.G*self.rho0*(self.Rs**3)*np.log(1+(r/self.Rs))/r

    def Ubulge(self, r):
        """
        Bulge contribution to potential divided by system mass
        """
        return -self.G*self.Mbulge/(r+self.abulge)

    def Vdot(self, time, COORD):
        """
        Vdot calculation using gradients of Ubulge, Udisk, and Uhalo
        """
        x,y,z = COORD[3:]
        xdot,ydot,zdot = COORD[:3]
        r = np.sqrt((x**2)+(y**2)+(z**2))

        def halo(self, r_i, r):
            """
            # gradient of Uhalo
            """
            term1 = r_i*np.log(1+(r/Rs))/(r**3)
            term2 = -r_i/((r**2)*(Rs+r))
            return self.G*4.0*np.pi*rho0*(Rs**3)*(term1+term2)


        def bulge_xy(self, r_i, r_j, z):
            """
            X and Y components of gradient of Ubulge

            i is component of interest, j is the other non-z component
            """
            return

        def bulge_z(self, x, y, z):
            """
            Z component of gradient of Ubulge

            """
            return

        def disk_xy(self, r_i, r_j, z):
            """
            X and Y components of gradient of Udisk

            i is component of interest, j is the other non-z component
            """
            return
        def disk_z(self, x, y, z):
            """
            Z component of gradient of Udisk
            """
            return


        gradUx = halo(x,r)+bulge_xy(x,y,z)+disk_xy(x,y,z)
        gradUy = halo(y,r)+bulge_xy(y,x,z)+disk_xy(y,x,z)
        gradUz = halo(z,r)+bulge_z(x,y,z)+disk_z(x,y,z)

        return np.array([-gradUx,-gradUy,-gradUz, xdot, ydot, zdot])
