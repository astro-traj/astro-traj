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
        Can specify methods 'gaussian' or 'delta_function' to sample using the mean and std of the posterior samples only
        """

        if not samples:
            raise ValueError("No posterior sample file specified!")
        
        samples = Table.read(samples, format='ascii')

        if method=='posterior':
            m1 = samples['m1'][np.random.randint(0,len(samples['m1']),size)]
            m2 = samples['m2'][np.random.randint(0,len(samples['m2']),size)]
            return m1, m2

        elif method=='delta_function':
            m1 = np.ones(size)*samples['m1'].mean()
            m2 = np.ones(size)*samples['m2'].mean()
            return m1, m2

        elif method=='gaussian':
            m1 = np.random.normal(samples['m1'].mean(), samples['m1'].std(), size)
            m2 = np.random.normal(samples['m2'].mean(), samples['m2'].std(), size)
            return m1, m2

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample semi-major axis
    def sample_Apre(self, Amin, Amax, size=None):
        '''
        samples semi-major axis uniformly
        '''
        A_samp = np.random.uniform(Amin, Amax, size)
        return A_samp


    # sample helium star mass
    def sample_Mhe(self, Mmin, Mmax=8.0, size=None):
        '''
        samples He-star mass uniformly between Mns and 8 Msun (BH limit)
        '''
        Mhe_samp = np.random.uniform(Mmin, Mmax, size=size)
        return Mhe_samp


    # sample kick velocities
    def sample_Vkick_maxwellian(self, scale=265, size=None):
        '''
        sample kick velocity from Maxwellian (Hobbs 2005)
        '''
        Vkick_samp = maxwell.rvs(loc=0, scale=scale, size=size)
        return Vkick_samp


    def sample_Vkick_uniform(self, Vmin=0.0, Vmax=2500.0, size=None):
        '''
        sample kick uniformly (Wong 2010)
        '''
        Vkick_samp = np.random.uniform(Vmin, Vmax, size=size)
        return Vkick_samp


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
