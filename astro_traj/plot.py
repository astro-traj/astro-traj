# -*- coding: utf-8 -*-
# Copyright (C) Michael Zevin (2017)
#
# This file is part of astro_traj
#
# gwemlightcurves is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwemlightcurves is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwemlightcurves.  If not, see <http://www.gnu.org/licenses/>.

"""`plotting`
"""
import constr_dict
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as C
from matplotlib import use
use('agg')
import galaxy
from sample import Sample
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.mlab import griddata
from scipy.stats import maxwell
import corner
import pdb

__author__ = 'Michael Zevin <michael.zevin@ligo.org>'
__all__ = ['Plotting']

class Plot:
    def __init__(self, samples, galaxy_name, offset, r_eff, telescope, output=None):
        '''
        initialize with values passed to gal and output file. outfile must be a string.
        '''
        # set cosmology to Reiss [h = H0/(100 km/s/Mpc)]
        h = 0.73
        # Info from "GW"
        GW = constr_dict.GW(samples)
        # Info about the galaxy
        Galaxy = constr_dict.galaxy(galaxy_name, samples, r_eff, offset, h)
        # Infer about the telescope that made the measurements (for angular resolution)
        tele = constr_dict.telescope(telescope)
        # Calculate angular resolution of the telescope, convert to physical size at the distance of the kilonova
        theta = 1.22 * 1e-9 * tele['lambda'] / tele['D']
        D_theta = Galaxy['d']*np.tan(theta)                # units of Mpc
        Galaxy['offset_uncer'] = D_theta*1000.0            # offset uncertainty due to angular resolution of telescope (kpc)
        # Initialize potential with galactic parameters, choose from one of the definied potentials in galaxy class
        gal=galaxy.Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], h, rcut=100)
        samp=Sample(gal)
        
        self.Galaxy = Galaxy
        self.tele = tele
        self.samp = samp
        self.Rpdf = samp.initialize_R()
        self.gal = gal    
        self.Mspiral = gal.Mspiral
        self.Mbulge = gal.Mbulge
        self.Mhalo = gal.Mhalo
        self.R_eff = gal.R_eff
        self.h = gal.h
        self.rcut = gal.rcut
        self.abulge = gal.abulge
        self.Rs = gal.Rs

        self.Ubulge = gal.Ubulge
        self.Uhalo = gal.Uhalo

        if output:
            outfile = str(output)+'.dat'
            data = pd.read_csv(outfile)
            self.data = data
    

    def cutTmerge(self,Tmerge,tmin,tmax = None):
        #tmin, tmax in Gyr
        
        if tmax:
            return np.where((Tmerge>tmin)&(Tmerge<tmax))[0]
        else:
            return np.where(Tmerge>tmin)[0]
        
    def getRhe(Mhe):
        if Mhe<=2.5:
            return 3.0965-(2.013*np.log10(Mhe))
        else:
            return 0.0557*((np.log10(Mhe)-0.172)**-2.5)
    def getRL(self,Mhe,M2,Apre):
        epre=np.full(Mhe.shape,0)
        
        exp1=1.0/3.0
        exp2=2.0/3.0
        q=Mhe/M2
        M=Mhe+M2 
        alpha=Apre
        ex=epre                         

        RL = (alpha*(1-ex)*0.49*(q**exp2))/((0.6*(q**exp2))+np.log(1.+(q**exp1)))
        return RL

        
    
    def cutOffset(self,Rmerge_proj,offset):
        #offset in kpc
        SigOffset = self.Galaxy['offset_uncer']
        #self.SigOffset(offset)
        
        return np.where((Rmerge_proj > offset-SigOffset) & (Rmerge_proj < offset  + SigOffset))[0]
    def BigOffset1D(self, filename, subject, xlabel, nOffset=[1.0 ,2./3.,2.0,3.0,5.0], norm = True,maxCut=None):
        from matplotlib import rcParams,ticker
        rcParams.update({'font.size': 18})
        
        offset = self.Galaxy['offset']
        Tmerge = self.data['Tmerge'].values
        Rmerge_proj = self.data['Rmerge_proj'].values.astype('float')
        flag = self.data['flag'].values
        fig = plt.figure(figsize = (40,20))

        ax0 = fig.add_subplot(231)
        axhalf = fig.add_subplot(232)
        ax1 = fig.add_subplot(233)
        ax2 = fig.add_subplot(234)
        axw1 = fig.add_subplot(235)
        axw2 = fig.add_subplot(236)
        
        if maxCut:
            ImaxCut =np.where(subject<maxCut)[0]
            Tmerge = Tmerge[ImaxCut]
            flag = flag[ImaxCut]
            Rmerge_proj = Rmerge_proj[ImaxCut]
            subject = subject[ImaxCut]
        Iwin = np.where(flag == 1)[0]
        Ihubble = np.where(flag != 2)[0]

        if norm:
            Constant = 1.0
            ylabel='PDF'
            alphaA=1.0
            alphaS = 0.5

        else:
            Constant = float(len(subject))
            ylabel = 'log(n)'
            ax0.set_yscale('log')
            axhalf.set_yscale('log')
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            axw1.set_yscale('log')
            axw2.set_yscale('log')

            alphaA=1.0
            alphaS = 0.5

        Amin,Amax = 0.1,10.0
        Mmin,Mmax = np.min(self.data['Mhe'].values),8.0

        ax0.set_title('Observed Offset ('+str(offset)+' kpc)')
        axhalf.set_title('Offset = '+str(np.round(nOffset[0],2))+'x Observed Offset')
        ax1.set_title('Offset = '+str(np.round(nOffset[1],2))+'x Observed Offset')
        ax2.set_title('Offset = '+str(nOffset[2])+'x Observed Offset')
        axw1.set_title('Offset = '+str(nOffset[3])+'x Observed Offset')
        axw2.set_title('Offset = '+str(nOffset[4])+'x Observed Offset')
        
        ax0.set_ylabel(ylabel)
        axhalf.set_ylabel(ylabel)
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        axw1.set_ylabel(ylabel)
        axw2.set_ylabel(ylabel)

        ax0.set_xlabel(xlabel)
        axhalf.set_xlabel(xlabel)
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        axw1.set_xlabel(xlabel)
        axw2.set_xlabel(xlabel)

        
        if xlabel == r'$A_{pre}$ in $R_{\odot}$':

            ax0.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            axhalf.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            ax1.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            ax2.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            axw1.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            axw2.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')

        if xlabel == r'$M_{He}$ in $M_{\odot}$':

            ax0.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            axhalf.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            ax1.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            ax2.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            axw1.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            axw2.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')

        if xlabel == '$V_{kick}$ in km/s':
            xx = np.linspace(0,max(subject),1000)

            ax0.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            axhalf.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            ax1.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            ax2.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            axw1.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            axw2.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')

        if xlabel == '$R_{birth}$ in kpc':
            xx = np.linspace(0,max(subject),1000)
            Rpdf=self.Rpdf.pdf

            ax0.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            axhalf.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            ax1.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            ax2.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            axw1.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            axw2.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')

        print (max(Tmerge[Ihubble]),max(Tmerge[Iwin]),max(Tmerge))
        
        I0 = Iwin
        Ihalf = np.intersect1d(self.cutOffset(Rmerge_proj,nOffset[0]*offset),Ihubble)
        I1 = np.intersect1d(self.cutOffset(Rmerge_proj,nOffset[1]*offset),Ihubble)
        I2 = np.intersect1d(self.cutOffset(Rmerge_proj,nOffset[2]*offset),Ihubble)
        Iw1 = np.intersect1d(self.cutOffset(Rmerge_proj,nOffset[3]*offset),Ihubble)
        Iw2 = np.intersect1d(self.cutOffset(Rmerge_proj,nOffset[4]*offset),Ihubble)

        nbins =int(np.sqrt(len(Iw2)))
        print nbins
        colorA = '.7'

        colorS = 'g'

        N, BINS, PATCHES = ax0.hist(subject[Ihubble], bins=nbins, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = axhalf.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = ax1.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = ax2.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = axw1.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = axw2.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')



        n, bins, patches = ax0.hist(subject[I0], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset constraints')
        n, bins, patches = axhalf.hist(subject[Ihalf], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset constraints')
        n, bins, patches = ax1.hist(subject[I1], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset constraints')
        n, bins, patches = ax2.hist(subject[I2], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset constraints')
        n, bins, patches = axw1.hist(subject[Iw1], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset constraints')
        n, bins, patches = axw2.hist(subject[Iw2], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset constraints')

        ax1.legend(loc=0)


        fig.savefig(filename)
   
    def Big1D(self, filename, subject, xlabel, TFLAG = [0.5,1,2], TWINDOW=[[1,2],[2,3]],norm = True,maxCut=None):
        from matplotlib import rcParams,ticker
        rcParams.update({'font.size': 18})

        Tmerge = self.data['Tmerge'].values
        flag = self.data['flag'].values
        fig = plt.figure(figsize = (40,20))

        ax0 = fig.add_subplot(231)
        axhalf = fig.add_subplot(232)
        ax1 = fig.add_subplot(233)
        ax2 = fig.add_subplot(234)
        axw1 = fig.add_subplot(235)
        axw2 = fig.add_subplot(236)
        
        if maxCut:
            Tmerge = Tmerge[np.where(subject<maxCut)[0]]
            flag = flag[np.where(subject<maxCut)[0]]
            subject = subject[np.where(subject<maxCut)[0]]
        Iwin = np.where(flag == 1)[0]
        Ihubble = np.where(flag != 2)[0]

        if norm:
            Constant = 1.0
            ylabel='PDF'
            alphaA=1.0
            alphaS = 0.5

        else:
            Constant = float(len(subject))
            ylabel = 'log(n)'
            ax0.set_yscale('log')
            axhalf.set_yscale('log')
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            axw1.set_yscale('log')
            axw2.set_yscale('log')

            alphaA=1.0
            alphaS = 0.5

        Amin,Amax = 0.1,10.0
        Mmin,Mmax = np.min(self.data['Mhe'].values),8.0

        ax0.set_title('All Systems')
        axhalf.set_title('Tmerge > '+str(TFLAG[0])+' Gyrs')
        ax1.set_title('Tmerge > '+str(TFLAG[1])+' Gyrs')
        ax2.set_title('Tmerge > '+str(TFLAG[2])+' Gyrs')
        axw1.set_title(str(TWINDOW[0][0])+' Gyrs > Tmerge > '+str(TWINDOW[0][1])+' Gyrs')
        axw2.set_title(str(TWINDOW[1][0])+' Gyrs > Tmerge > '+str(TWINDOW[1][1])+' Gyrs')
        
        ax0.set_ylabel(ylabel)
        axhalf.set_ylabel(ylabel)
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        axw1.set_ylabel(ylabel)
        axw2.set_ylabel(ylabel)

        ax0.set_xlabel(xlabel)
        axhalf.set_xlabel(xlabel)
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        axw1.set_xlabel(xlabel)
        axw2.set_xlabel(xlabel)

        
        if xlabel == r'$A_{pre}$ in $R_{\odot}$':

            ax0.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            axhalf.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            ax1.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            ax2.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            axw1.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')
            axw2.axhline(Constant/(Amax-Amin),color='g',label = 'Input Distribution')

        if xlabel == r'$M_{He}$ in $M_{\odot}$':

            ax0.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            axhalf.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            ax1.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            ax2.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            axw1.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')
            axw2.axhline(Constant/(Mmax-Mmin),color='g',label = 'Input Distribution')

        if xlabel == '$V_{kick}$ in km/s':
            xx = np.linspace(0,max(subject),1000)

            ax0.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            axhalf.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            ax1.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            ax2.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            axw1.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
            axw2.plot(xx,Constant*maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')

        if xlabel == '$R_{birth}$ in kpc':
            xx = np.linspace(0,max(subject),1000)
            Rpdf=self.Rpdf.pdf

            ax0.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            axhalf.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            ax1.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            ax2.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            axw1.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')
            axw2.plot(xx,Constant*Rpdf(xx),'g-',label='Input Distribution')

        print (max(Tmerge[Ihubble]),max(Tmerge[Iwin]),max(Tmerge))
        
        I0 = Iwin
        Ihalf = np.intersect1d(self.cutTmerge(Tmerge,TFLAG[0]),Iwin)
        I1 = np.intersect1d(self.cutTmerge(Tmerge,TFLAG[1]),Iwin)
        I2 = np.intersect1d(self.cutTmerge(Tmerge,TFLAG[2]),Iwin)
        Iw1 = np.intersect1d(self.cutTmerge(Tmerge,TWINDOW[0][0],tmax=TWINDOW[0][1]),Iwin)
        Iw2 = np.intersect1d(self.cutTmerge(Tmerge,TWINDOW[1][0],tmax=TWINDOW[1][1]),Iwin)

        nbins = int(np.sqrt(len(Iw2)))
        
        colorA = '.7'

        colorS = 'r'

        N, BINS, PATCHES = ax0.hist(subject[Ihubble], bins=nbins, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = axhalf.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = ax1.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = ax2.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = axw1.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')
        n, bins, patches = axw2.hist(subject[Ihubble], bins=BINS, normed = norm, color = colorA,alpha = alphaA,label='SN Survivors Merging within Hubble Time')



        n, bins, patches = ax0.hist(subject[I0], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset and Tmerge constraints')
        n, bins, patches = axhalf.hist(subject[Ihalf], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset and Tmerge constraints')
        n, bins, patches = ax1.hist(subject[I1], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset and Tmerge constraints')
        n, bins, patches = ax2.hist(subject[I2], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset and Tmerge constraints')
        n, bins, patches = axw1.hist(subject[Iw1], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset and Tmerge constraints')
        n, bins, patches = axw2.hist(subject[Iw2], bins=BINS, normed=norm, color=colorS, alpha=alphaS, label = 'Matches to offset and Tmerge constraints')

        ax1.legend(loc=0)


        fig.savefig(filename)
   
    def input1Dpdf(self,filename='other.png',tflag=None):
        from scipy.stats import rv_continuous
        from matplotlib import rcParams,ticker
        fig = plt.figure(figsize=(40,20))

        rcParams.update({'font.size': 18})
        Rpdf=self.Rpdf.pdf
        
        D=self.data
        Apre = D['Apre'].values
        Mhe = D['Mhe'].values
        Vkick = D['Vkick'].values
        R = D['R'].values
        
        print max(R)
        galphi = D['galphi'].values
        galcosth = D['galcosth'].values
        if tflag:
            t=D['Tmerge'].values*u.year.to(u.Gyr)
            I=np.where((t>tflag) & (t<tflag+1.0))[0]
            fig.suptitle('For 2.5>$T_{merge}$ > 1.5 Gyr')
            Apre,Mhe,Vkick,R,galphi,galcosth = Apre[I], Mhe[I], Vkick[I],R[I],galphi[I],galcosth[I]
        
        Amin,Amax = 0.1,10.0
        Mmin,Mmax = np.min(Mhe),8.0
        print 'r'
        print R
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)
        ax1.set_xlabel(r'$M_{He}$ in $M_{\odot}$')
        ax2.set_xlabel(r'$A_{preSN}$ in $R_{\odot}$')
        ax3.set_xlabel('$V_{kick}$ in km/s')
        ax4.set_xlabel('$R_{birth}$ in kpc')
        ax5.set_xlabel('$\phi$')
        ax6.set_xlabel(r'$cos(\theta)$')

        ax1.set_ylabel('PDF')
        ax2.set_ylabel('PDF')
        ax3.set_ylabel('PDF')
        ax4.set_ylabel('PDF')
        ax5.set_ylabel('PDF')
        ax6.set_ylabel('PDF')
        nbins=40#int(np.sqrt(len(Apre)))
        print nbins
        n,bins,patches = ax1.hist(Mhe,bins=nbins, normed=True,color='.7')
        ax1.axhline(1.0/(Mmax-Mmin),color='g',label='Input Distribution')
        ax1.legend(loc=0)
        
        n,bins,patches = ax2.hist(Apre, bins=nbins,normed=True, color='.7')
        ax2.axhline(1.0/(Amax-Amin),color='g',label='Input Distribution')
        ax2.legend(loc=0)

        n,bins,patches = ax3.hist(Vkick, bins=nbins,normed=True,color='.7')
        xx = np.linspace(0,1000,1000)
        ax3.plot(xx,maxwell.pdf(xx,loc=0,scale=265.0),'g-',label='Input Distribution')
        ax3.legend()


        n,bins,patches = ax4.hist(R, bins=nbins,normed=True,color='.7')
        xx = np.linspace(0,20,1000)
        ax4.plot(xx,Rpdf(xx),'g-',label='Input Distribution')
        ax4.legend()
        
        n,bins,patches = ax5.hist(galphi, bins=nbins,normed=True,color='.7')
        #ax5.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
        #ax5.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        ax5.axhline(1.0/(2*np.pi),color='g',label='Input Distribution')
        ax5.legend()
        
        n,bins,patches = ax6.hist(galcosth,bins=nbins, normed=True,color='.7')
        ax6.axhline(0.5,color='g',label='Input Distribution')
        ax6.legend()
        print filename
        print self
        fig.savefig(filename)

    def output1Dpdf(self,filename='other.png',tflag=None):
        from scipy.stats import rv_continuous
        from matplotlib import rcParams,ticker
        fig = plt.figure(figsize=(40,20))

        rcParams.update({'font.size': 18})


  
        
        D=self.data
        Apost = D['Apost'].values
        epost = D['epost'].values
        Tmerge = D['Tmerge'].values*u.year.to(u.Myr)
        #Rmerge = D['R'].values
        #Vfinal = D['Vfinal'].values

        if tflag:
            t=D['Tmerge'].values*u.year.to(u.Gyr)
            I=np.where((t>tflag) & (t<tflag+1.0))[0]
            fig.suptitle('For 2.5 > $T_{merge}$ > 1.5 Gyr')
            #Apost,epost,Vfinal,Rmerge,Tmerge = Apost[I], epost[I], Vfinal[I],Rmerge[I],Tmerge[I]
            Apost,epost,Tmerge=Apost[I],epost[I],Tmerge[I]
        nbins=40
        print len(Apost)
        Apost = Apost[np.where(Apost<20)[0]]
        Rmerge = Rmerge[np.where(Rmerge<15)[0]]
        Tmerge=Tmerge*np.log10(Tmerge)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        #ax3 = fig.add_subplot(233)
        #ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(133)
        ax1.set_xlabel(r'$A_{postSN}$ in $R_{\odot}$')
        ax2.set_xlabel('$e_{postSN}$')
        #ax3.set_xlabel('$V_{final}$ in km/s')
        #ax4.set_xlabel('$R_{merge}$ in kpc')
        ax5.set_xlabel('log10($T_{merge}$/Myr)')

        ax1.set_ylabel('PDF')
        ax2.set_ylabel('PDF')
        ax3.set_ylabel('PDF')
        ax4.set_ylabel('PDF')
        ax5.set_ylabel('PDF')
        print nbins
        n,bins,patches = ax1.hist(Apost,bins=nbins, normed=True,color='.7')
        
        n,bins,patches = ax2.hist(epost, bins=nbins,normed=True, color='.7')

        #n,bins,patches = ax3.hist(Vfinal, bins=nbins,normed=True,color='.7')

        #n,bins,patches = ax4.hist(Rmerge, bins=nbins,normed=True,color='.7')
        
        n,bins,patches = ax5.hist(Tmerge, bins=nbins,normed=True,color='.7')

        

        fig.savefig(filename)


        


        

    def potential(self):
        f, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
        rsamp = np.linspace(0.0001,100,10000) * C.kpc.value        
        axs.plot(rsamp/C.kpc.value, np.abs(self.Ubulge(rsamp)), color='blue', label='Stellar Bulge')
        axs.plot(rsamp/C.kpc.value, np.abs(self.Uhalo(rsamp)), color='red', label='DM Halo')
        axs.plot(rsamp/C.kpc.value, np.abs(self.Ubulge(rsamp)+self.Uhalo(rsamp)), color='magenta', label='Combined Potential')
        axs.set_yscale('log')
        axs.set_xlabel('radius (kpc)')
        axs.set_ylabel('|V(r)|')
        axs.axvline(self.abulge/C.kpc.value, color='k', linestyle=':', label='abulge')
        axs.axvline(self.Rs/C.kpc.value, color='k', linestyle='--', label='Rs')
        axs.set_xlim(0,100)
        plt.legend()
        plt.savefig('potential.png')


    def postSN_traj(self, X, Y, Z):
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(X,Y,Z,'g-',label='PostSN Trajectory')
        ax.plot([X[0]],[Y[0]],[Z[0]],'ro')
        plt.savefig('postSN_traj.png')



    def Vsys_hist(self, Vsystems):
        f, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
        axs.hist(Vsystems,bins=np.logspace(10,1000,40))
        xx=np.linspace(0,1000,1000)
        axs.semilogx(xx,maxwell.pdf(xx,loc=0,scale=265))

        axs.set_xlabel('$V_{sys}$ Post SN km/s')
        axs.set_ylabel('pdf')
        plt.savefig('Vsys_hist.png')


    def corner_plot(self, tlow, thigh):
        params = ['Vkick', 'Mhe', 'Apre', 'Apost', 'R', 'Rmerge', 'Tmerge']
        reduced_df = self.data[params]
        if tlow:
            reduced_df = reduced_df[reduced_df['Tmerge'] > tlow]
        if thigh:
            reduced_df = reduced_df[reduced_df['Tmerge'] < thigh]
        reduced = np.asarray(reduced_df)
        corner.corner(reduced, bins=25, labels=params, show_titles=True, quantiles=[0.1,0.9], range=[.99]*len(params), plot_contours=True)
        plt.savefig('corner.png')


    def Vkick_Mhe_pdf(self, x_param='Vkick', y_param='Mhe', c_param=None):
        fig = plt.figure()

        gs = gridspec.GridSpec(4,4)

        x = self.data[x_param]
        y = self.data[y_param]
        if c_param:
            z = self.data[c_param] / 1e9

        x_grid = np.linspace(x.min(), x.max(), 1000)
        y_grid = np.linspace(y.min(), y.max(), 1000)

        ax_joint = fig.add_subplot(gs[1:4,0:3])
        ax_marg_x = fig.add_subplot(gs[0,0:3])
        ax_marg_y = fig.add_subplot(gs[1:4,3])

        ax_joint.scatter(x,y, c=z, cmap='viridis')
        ax_marg_x.hist(x)
        ax_marg_y.hist(y,orientation="horizontal")

        # Turn off tick labels on marginals
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Set labels on joint
        ax_joint.set_xlabel(x_param)
        ax_joint.set_ylabel(y_param)
    
        plt.savefig('vkick_mhe_pdf.png')
