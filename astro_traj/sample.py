# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
#
# This file is part of astro-traj.
#
# astro-traj is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# astro-traj is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with astro-traj.  If not, see <http://www.gnu.org/licenses/>.

"""`sample`
"""

import numpy as np
import astropy.units as u
import astropy.constants as C
from scipy.stats import maxwell
from scipy.integrate import trapz
from scipy.stats import rv_continuous
from astropy.table import Table
__author__ = ['Chase Kimball <charles.kimball@ligo.org>', 'Michael Zevin <michael.zevin@ligo.org>']
__credits__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['Sample', 'Hernquist_pdf']

class Hernquist_pdf(rv_continuous):
    '''
    density pdf from Hernquist potential
    '''
    def __init__(self, abulge, rcut, momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):
        rv_continuous.__init__(self, momtype, a, b, xtol,
                 badvalue, name, longname,
                 shapes, extradoc, seed)
        self.abulge = abulge
        self.rcut = rcut


    def _pdf(self, r):
        # Divided by Mbulge. Turns mass density function to probability density function.
        # rvs() requires _pdf normalized to 1 on domain, which in this case is [0,inf]
        # Leaving Mbulge in there normalizes to Mbulge. Check normalization by printing Hernquist_pdf.cdf(big number)
        # WARNING: Hernquist_pdf.cdf(np.inf) will always return 1 because it assumes we normalized correctly
        def normFac(n):
            if n ==0:
                return 1.0
            else:
                return ((n+1)**2)/(n**2)
        Cnorm = normFac(self.rcut/self.abulge)            # NOTE: rcut needs to be in terms of abulge for normalization
        return 2 * Cnorm*r * self.abulge * (self.abulge + r)**(-3)


class Sample:
    def __init__(self, gal):   # default rcut=0 does not truncate the distributions (i.e., they extend to infinity)
        '''
        initialize with values passed to gal
        '''
        self.abulge = gal.abulge / C.kpc.value
        self.rcut = gal.rcut / C.kpc.value


    # sample compact binary masses from PE
    def sample_masses(self, samples=None, method='posterior', size=None):
        """
        Samples m1 and m2 from posterior distrbution of your favorite PE run.
        Samples from the posterior samples by default. 
        Can specify methods 'gaussian', 'mean', or 'median' to sample using other sampling methods
        """

        if not samples:
            raise ValueError("No posterior sample file specified!")
        
        samples = Table.read(samples, format='ascii')

        if method=='posterior':
            m1 = samples['m1_source'][np.random.randint(0,len(samples['m1_source']),size)]
            m2 = samples['m2_source'][np.random.randint(0,len(samples['m2_source']),size)]
            return m1, m2

        elif method=='mean':
            m1 = np.ones(size)*samples['m1_source'].mean()
            m2 = np.ones(size)*samples['m2_source'].mean()
            return m1, m2

        elif method=='median':
            m1 = np.ones(size)*np.median(samples['m1_source'])
            m2 = np.ones(size)*np.median(samples['m2_source'])
            return m1, m2

        elif method=='gaussian':
            m1 = np.random.normal(np.median(samples['m1_source']), samples['m1_source'].std(), size)
            m2 = np.random.normal(np.median(samples['m2_source']), samples['m2_source'].std(), size)
            return m1, m2

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample distance from PE
    def sample_distance(self, samples=None, method='median', size=None):
        """
        Samples distance from posterior distrbution of your favorite PE run.
        Just uses the mean value for distance by default. 
        Can specify methods 'gaussian', 'mean', or 'posteriors' to sample using other methods
        """

        if not samples:
            raise ValueError("No posterior sample file specified!")
        
        samples = Table.read(samples, format='ascii')

        if method=='posterior':
            d = samples['distance'][np.random.randint(0,len(samples['distance']),size)]
            return d

        elif method=='mean':
            d = np.ones(size)*samples['distance'].mean()
            return d

        elif method=='median':
            d = np.ones(size)*np.median(samples['distance'])
            return d

        elif method=='gaussian':
            d = np.random.normal(np.median(samples['distance']), samples['distance'].std(), size)
            return d

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample semi-major axis
    def sample_Apre(self, Amin, Amax, method='uniform', size=None):
        '''
        samples semi-major axis uniformly (method='uniform', default) or uniformly in log (method='log')
        '''
        if method=='uniform':
            A_samp = np.random.uniform(Amin, Amax, size)
            return A_samp

        elif method=='log':
            A_samp = 10**np.random.uniform(np.log10(Amin), np.log10(Amax), size)
            return A_samp

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample eccentricity
    def sample_epre(self, method='circularized',  size=None):
        '''
        samples initial eccentricity (for now, assume circularized)
        '''
        if method=='circularized':
            e_samp = np.zeros(size)
            return e_samp

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample helium star mass
    def sample_Mhe(self, Mmin, Mmax=8.0, method='uniform', size=None):
        '''
        samples He-star mass uniformly between Mns and 8 Msun (BH limit)
        '''
        if method=='power':
            Mmin=2.
            def pdf(m):
                return m**-2.3
            xx=np.linspace(Mmin,Mmax,1000)
            A1=trapz(pdf(xx),x=xx)
            Anorm=1./A1
            def invpdf(ii):
                return (1./((Mmin**-1.3)-(ii*1.3/Anorm)))**(1./1.3)
            II=np.random.uniform(0,1,size=size)
            return invpdf(II)

            

        if method=='uniform':
            Mhe_samp = np.random.uniform(Mmin, Mmax, size=size)
            return Mhe_samp

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample kick velocities
    def sample_Vkick(self, scale=265, Vmin=0, Vmax=2500, method='maxwellian', size=None):
        '''
        sample kick velocity from Maxwellian (Hobbs 2005, default) or uniformly (Wong 2010)
        '''
        if method=='maxwellian':
            Vkick_samp = maxwell.rvs(loc=0, scale=scale, size=size)
            return Vkick_samp

        elif method=='uniform':
            Vkick_samp = np.random.uniform(Vmin, Vmax, size=size)
            return Vkick_samp

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    def initialize_R(self):
        '''
        samples radial distance from galactic center according to specified potential function
        '''

        if self.rcut == 0:
            return Hernquist_pdf(abulge=self.abulge, rcut=self.rcut, a=0,name='my_pdf')
        else:
            return Hernquist_pdf(abulge=self.abulge, rcut=self.rcut, a=0, b=self.rcut, name='my_pdf')


    def sample_R(self, PDF, Ndraws):
        '''
        samples radial distance from galactic center according to specified potential function
        '''
        return PDF.rvs(size=Ndraws)
