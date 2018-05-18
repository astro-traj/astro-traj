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
__all__ = ['Sample', 'Hernquist_pdf', 'BeniaminiKick_pdf', 'BeniaminiMhe_pdf']

import numpy as np
import astropy.units as units

from scipy.stats import maxwell
from scipy.integrate import trapz
from scipy.stats import rv_continuous
from astropy.table import Table

class Hernquist_pdf(rv_continuous):
    '''
    density pdf from Hernquist potential
    '''
    def __init__(self, abulge, momtype=1, a=0, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):
        rv_continuous.__init__(self, momtype, a, b, xtol,
                 badvalue, name, longname,
                 shapes, extradoc, seed)
        self.abulge = abulge
        self.Rmax = b

    def _pdf(self, r):
        # Divided by Mbulge. Turns mass density function to probability density function.
        # rvs() requires _pdf normalized to 1 on domain, which in this case is [0,inf]
        # Leaving Mbulge in there normalizes to Mbulge. Check normalization by printing Hernquist_pdf.cdf(big number)
        # WARNING: Hernquist_pdf.cdf(np.inf) will always return 1 because it assumes we normalized correctly
        def normFac(n):
            return ((n+1)**2)/(n**2)
        # see if a maximum sampling radius has been specified
        if self.Rmax:
            Cnorm = normFac(self.Rmax/self.abulge)            # NOTE: Rmax needs to be in terms of abulge for normalization
        else:
            Cnorm = 1
        return 2 * Cnorm*r * self.abulge * (self.abulge + r)**(-3)

class BeniaminiKick_pdf(rv_continuous):
    '''
    vkick pdf from Beniamini & Piran 2016
    '''
    def __init__(self,Vk0,sigvk = np.arcsinh(.5),momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):
        
        rv_continuous.__init__(self, momtype, a, b, xtol,
                 badvalue, name, longname,
                 shapes, extradoc, seed)
        self.Vk0 = Vk0
        self.sigvk = sigvk

    def _pdf(self,Vk):
            term1 = 1/(np.sqrt(2.0*np.pi)*self.sigvk*Vk)
            term2 = np.exp(-(np.log(Vk/self.Vk0)**2)/(2.0*(self.sigvk**2)))
            return term1*term2
        
class BeniaminiMhe_pdf(rv_continuous):
    '''
    vkick pdf from Beniamini & Piran 2016
    '''
    def __init__(self,dM0,sigdM = np.arcsinh(.5),momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):
        
        rv_continuous.__init__(self, momtype, a, b, xtol,
                 badvalue, name, longname,
                 shapes, extradoc, seed)
        self.dM0 = dM0
        self.sigdM = sigdM

    def _pdf(self,dM):
            term1 = 1/(np.sqrt(2.0*np.pi)*self.sigdM*dM)
            term2 = np.exp(-(np.log(dM/self.dM0)**2)/(2.0*(self.sigdM**2)))
            return term1*term2 


class Sample:
    def __init__(self, NS, gal, Rmax=None):   # default rcut=0 does not truncate the distributions (i.e., they extend to infinity)
        '''
        initialize with values passed to gal
        NOTE: gal contains parameter values with SI units
        '''
        self.abulge = gal.abulge * units.m.to(units.kpc)
        if Rmax:
            self.Rmax = Rmax * units.m.to(units.kpc)
        else:
            # if Rmax not specified maintain it as None
            self.Rmax = None

        self.m1 = NS['m1']
        self.m2 = NS['m2']
        self.m1_sigma = NS['m1_sigma']
        self.m2_sigma = NS['m2_sigma']


    # sample compact binary masses from PE
    def sample_masses(self, method='gaussian', samples=None, size=None):
        """
        Samples m1 and m2 from posterior distrbution of your favorite PE run.
        Samples from the posterior samples by default. 
        Can specify methods 'gaussian', 'mean', or 'median' to sample using other sampling methods
        """

        if samples:
            samples = Table.read(samples, format='ascii')

        if method=='posterior':
            m1 = samples['m1_source'][np.random.randint(0,len(samples['m1_source']),size)]
            m2 = samples['m2_source'][np.random.randint(0,len(samples['m2_source']),size)]
            return m1, m2

        elif method=='mean':
            m1 = sefl.m1
            m2 = self.m2
            return m1, m2

        elif method=='median':
            m1 = np.ones(size)*np.median(samples['m1_source'])
            m2 = np.ones(size)*np.median(samples['m2_source'])
            return m1, m2

        elif method=='gaussian':
            m1 = np.random.normal(self.m1, self.m1_sigma, size)
            m2 = np.random.normal(self.m2, self.m2_sigma, size)
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
    def initialize_Mhe(self,dM0):
        return BeniaminiMhe_pdf(dM0,a=0)
    def sample_Mhe(self, Mns, Mmax=8.0, method='uniform', PDF=None, ECSN_PDF=None, CCSN_PDF=None, irand=None):
        '''
        samples He-star mass uniformly between Mns and 8 Msun (BH limit): beniamini2 method selects from two 
        distributions ECS and CCSN. The split is based off the 60/40 split observed in double nurtron stars 
        in our galaxy laid out in Fig 2: https://arxiv.org/pdf/1510.03111.pdf#figure.2 method: powerlaw
        '''

        if method=='uniform':
            Mhe_samp=[]

            for Mmin in Mns:
                Mhe_samp.append(np.random.uniform(Mmin, Mmax))
            return np.asarray(Mhe_samp)

        if method=='power':
            Mhe_samp=[]
            
            def pdf(m):
                return m**-2.35

            def invpdf(ii,m):
                    return (1./((m**-1.3)-(ii*1.3/Anorm)))**(1./1.3)
            for Mmin in Mns:
                xx=np.linspace(Mmin,Mmax,1000)
                A1=trapz(pdf(xx),x=xx)
                Anorm=1./A1

                II=np.random.uniform(0,1)
                Mhe_samp.append(invpdf(II,Mmin))
            return np.array(Mhe_samp)
            
        if method=='beniamini_1pop':
            dMhe_samp = []
            for Mmin in Mns:
                dMhe_samp.append(PDF.rvs())
            return np.asarray(dMhe_samp)+Mmin

        if method=='beniamini_2pop':
            dMhe_samp = []
            for Mmin in Mns:
                if dumrand[i]<0.6:
                    dMhe_samp.append(ECSPDF.rvs())
                else:
                    dMhe_samp.append(CCSPDF.rvs())
            return np.array(dMhe_samp)+Mmin
        
        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # sample kick velocities
    def initialize_Vkick(self):
        ECSN = BeniaminiKick_pdf(5.0,a=0)
        CCSN = BeniaminiKick_pdf(158.0,a=0)
        return ECSN, CCSN
    def sample_Vkick(self, scale=265, Vmin=0, Vmax=2000, method='maxwellian', size=None, Mhe=None, ECSN_PDF=None, CCSN_PDF=None, irand=None):
        '''
        sample kick velocity from Maxwellian (Hobbs 2005, default) or uniformly (Wong 2010) or Beniamini (2016): https://arxiv.org/pdf/1510.03111.pdf#equation.4.7: beniamini2 method selects from two distributions ECS and CCSN. The splitis based off the 60/40 split observed in double nurtron stars in our galaxy laid out in Fig 2: https://arxiv.org/pdf/1510.03111.pdf#figure.2
        '''
        if method=='beniamini_1pop':
            Vkick_samp=[]
            for i in range(len(Mhe)):
                if Mhe[i]<=2.25:
                    Vkick_samp.append(ECSN_PDF.rvs())
                else:
                    Vkick_samp.append(CCSN_PDF.rvs())
            return np.array(Vkick_samp)

        if method=='beniamini_2pop':
            Vkick_samp=[]
            for i in range(len(irand)):
                if irand[i]<=0.6:
                    Vkick_samp.append(ECSN_PDF.rvs())
                else:
                    Vkick_samp.append(CCSN_PDF.rvs())
            return np.array(Vkick_samp)
                    
            
        if method=='maxwellian':
            Vkick_samp = maxwell.rvs(loc=0, scale=scale, size=size)
            return Vkick_samp

        elif method=='uniform':
            Vkick_samp = np.random.uniform(Vmin, Vmax, size=size)
            return Vkick_samp

        else: 
            raise ValueError("Undefined sampling method: %s" % method)


    # Sample distance from galaxy center
    def initialize_R(self):
        '''
        samples radial distance from galactic center according to specified potential function
        '''

        if self.Rmax:
            return Hernquist_pdf(abulge=self.abulge, b=self.Rmax, name='my_pdf')
        else:
            return Hernquist_pdf(abulge=self.abulge, name='my_pdf')


    def sample_R(self, PDF, Ndraws):
        '''
        samples radial distance from galactic center according to specified potential function
        '''
        return PDF.rvs(size=Ndraws)



