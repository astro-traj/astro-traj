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
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import maxwell
import corner
import pdb

__author__ = 'Michael Zevin <michael.zevin@ligo.org>'
__all__ = ['Plotting']

class Plot:
    def __init__(self, samples, galaxy_name, offset, r_eff, telescope, output=None, flatten=False):
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

        self.gal = gal
        self.Ubulge = gal.Ubulge
        self.Uhalo = gal.Uhalo

        if output:
            outfile = output
            data = pd.read_csv(outfile)
            self.data = data

        if flatten:
            scale = 5.0
            from scipy.integrate import quad
            # integrate over z-axis
            def rho(x,y,z):
               return 2 * np.sqrt(x**2 + y**2 + z**2) * self.abulge/C.kpc.value * (self.abulge/C.kpc.value + np.sqrt(x**2 + y**2 + z**2))**(-3)
            def rho_flat(x,y):
                return quad(rho, -np.inf, np.inf, args=(x,y))[0]

            x = np.linspace(-1*scale, scale, 100)
            y = np.linspace(-1*scale, scale, 100)
            xx, yy = np.meshgrid(x,y)
            
            col=[]
            for i in xrange(len(xx.flatten())): col.append(rho_flat(xx.flatten()[i], yy.flatten()[i]))
            col=np.asarray(col)
            self.col = col
            return

    def getSigOffset(self,d):
        
        #d in kpc#
        theta = 1.22 * 1e-9 * self.tele['lambda'] / self.tele['D']
        D_theta = self.Galaxy['d']*np.tan(theta)                # units of Mpc
        SigOffset = D_theta*1000.0            # offset uncertainty due to angular resolution of telescope (kpc)
        return SigOffset

    def cutTmerge(self,tmin,tmax = None):
        #tmin, tmax in Gyr
        tmin = tmin*u.Gyr.to(u.yr)
        if tmax:
            tmax = tmax*u.Gyr.to(u.yr)
            return np.where((self.data['Tmerge'].values>tmin)&(self.data['Tmerge'].values<tmax))[0]

    def cutOffset(self,offset):
        #offset in kpc
        SigOffset = self.SigOffset(offset)
        Rmerge = self.data['Rmerge_proj'].values #kpc
        return np.where((Rmerge > offset-SigOffset) & (Rmerge < offset  + SigOffset))[0]
    
        
    def Tpdf(self,filename = 'other.png',TFLAG=None,TWINDOW=None):
        from scipy.stats import rv_continuous
        from matplotlib import rcParams, ticker
        fig = plt.figure(figsize = (40,20))
        rcParams.update({'font.size':18})

        
        Rpdf=self.Rpdf.pdf
        D=self.data
        Apre = D['Apre'].values
        Mhe = D['Mhe'].values
        Vkick = D['Vkick'].values
        R = D['R'].values
        Apost = D['Apost'].values
        epost = D['epost'].values
        #Rmerge = D['Rmerge'].values
        #Vfinal = D['Vfinal'].values
        Tmerge = D['Tmerge'].values*u.year.to(u.Myr)
        
        def cutTflag(tflag):
            t=D['Tmerge'].values*u.year.to(u.Gyr)
            I=np.where(t>tflag)[0]
            return Apre[I],Mhe[I],Vkick[I],R[I],Apost[I],epost[I],Tmerge[I]#,Rmerge[I],Vfinal[I]
        def cutTwindow(twindow):
            t=D['Tmerge'].values*u.year.to(u.Gyr)
            I=np.where((t>twindow[0])&(t<twindow[1]))[0]
            return Apre[I],Mhe[I],Vkick[I],R[I],Apost[I],epost[I],Tmerge[I]#,Rmerge[I],Vfinal[I]
        
        Aprehalf,Mhehalf,Vkickhalf,Rhalf,Aposthalf,eposthalf,Tmergehalf = cutTflag(TFLAG[0])
        Apre1,Mhe1,Vkick1,R1,Apost1,epost1,Tmerge1 = cutTflag(TFLAG[1])
        Apre2,Mhe2,Vkick2,R2,Apost2,epost2,Tmerge2= cutTflag(TFLAG[2])

        Aprew0,Mhew0,Vkickw0,Rw0,Apostw0,epostw0,Tmergew0 = cutTwindow(TWINDOW[0])
        Aprew1,Mhew1,Vkickw1,Rw1,Apostw1,epostw1,Tmergew1 = cutTwindow(TWINDOW[1])

    
        ax0 = fig.add_subplot(231)
        axhalf = fig.add_subplot(232)
        ax1 = fig.add_subplot(233)
        ax2 = fig.add_subplot(234)
        axw0 = fig.add_subplot(235)
        axw1 = fig.add_subplot(236)


        
        #r'$A_{preSN}$ in $R_{\odot}$')
        #'$V_{kick}$ in km/s')
        #'$R_{birth}$ in kpc')


        ax0.set_ylabel('PDF')
        axhalf.set_ylabel('PDF')
        ax1.set_ylabel('PDF')
        ax2.set_ylabel('PDF')
        axw0.set_ylabel('PDF')
        axw1.set_ylabel('PDF')

        ax0.set_title('All Systems')
        axhalf.set_title('Tmerge > '+str(TFLAG[0])+' Gyrs')
        ax1.set_title('Tmerge > '+str(TFLAG[1])+' Gyrs')
        ax2.set_title('Tmerge > '+str(TFLAG[2])+' Gyrs')
        axw0.set_title(str(TWINDOW[0][0])+' Gyrs > Tmerge > '+str(TWINDOW[0][1])+' Gyrs')
        axw1.set_title(str(TWINDOW[1][0])+' Gyrs > Tmerge > '+str(TWINDOW[1][1])+' Gyrs')
        nbins = 40

        xlabel = '$e_{post}$'

        N, BINS, PLOTS = ax0.hist(epost,bins = nbins, normed = True, color = '.7')

        N,bins, PLOTS = axhalf.hist(eposthalf,bins = BINS, normed = True, color = '.7')
        N,bins, PLOTS = ax1.hist(epost1,bins = BINS, normed = True, color = '.7')
        N,bins, PLOTS = ax2.hist(epost2,bins = BINS, normed = True, color = '.7')
        N,bins, PLOTS = axw0.hist(epostw0,bins = BINS, normed = True, color = '.7')
        N,bins, PLOTS = axw1.hist(epostw1,bins = BINS, normed = True, color = '.7')
        fig.suptitle('All Bound PostSN Systems')
        

    



        ax0.set_xlabel(xlabel)
        axhalf.set_xlabel(xlabel)
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        axw0.set_xlabel(xlabel)
        axw1.set_xlabel(xlabel)

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




    def trajectory(self, traj_file):
        df = pd.read_csv(traj_file+'.dat')
        dfi = pd.read_csv(traj_file+'_ini.dat')
        num = traj_file[10:]

        # setup axes
        fig = plt.figure()
        gs = gridspec.GridSpec(8,9)
        ax_gal = fig.add_subplot(gs[:,:-1])
        ax_cbar = fig.add_subplot(gs[:,-1])

        # plot stellar background
        def norm(x):
            return (x-x.min())/(x.max()-x.min())
        col_norm = norm(self.col)
        scale = 5.0
        x = np.linspace(-1*scale, scale, 100)
        y = np.linspace(-1*scale, scale, 100)
        xx, yy = np.meshgrid(x,y)

        ax_gal.scatter(xx, yy, c=col_norm, cmap='Greys', label=None, zorder=1)


        # plot pre-SN circular orbit
        from scipy.integrate import ode
        Ri = np.sqrt(df.iloc[0]['x']**2+df.iloc[0]['y']**2+df.iloc[0]['z']**2)
        Vi = np.sqrt(dfi.iloc[0]['vx']**2+dfi.iloc[0]['vy']**2+dfi.iloc[0]['vz']**2)
        Torb = 2*np.pi*Ri/Vi
        solver=ode(self.gal.Vdot).set_integrator('dopri5', nsteps=1e13, max_step=u.year.to(u.s)*1e6, rtol=1e-11)
        sol = []
        def solout(t,y):
            temp=list(y)
            temp.append(t)
            sol.append(temp)
        RR = np.array([dfi.iloc[0]['vx'],dfi.iloc[0]['vy'],dfi.iloc[0]['vz'],df.iloc[0]['x'],df.iloc[0]['y'],df.iloc[0]['z']])
        solver.set_solout(solout)
        solver.set_initial_value(RR,0)
        solver.integrate(Torb)
        sol=np.array(sol)
        
        ax_gal.plot(sol[:,3]*u.m.to(u.kpc), sol[:,4]*u.m.to(u.kpc), color='g', label='$Initial\ Orbit$', zorder=3, alpha=0.7)

        # plot the trajectory of the binary through the galaxy, colored by age
        age_col = df['t']*u.s.to(u.Gyr)
        pts = ax_gal.scatter(df['x']*u.m.to(u.kpc), df['y']*u.m.to(u.kpc), c=age_col, cmap='viridis', s=0.5, zorder=2, label=None, alpha=0.5)

        # plot location of supernova and kilonova
        ax_gal.scatter(df.iloc[0]['x']*u.m.to(u.kpc), df.iloc[0]['y']*u.m.to(u.kpc), marker='*', color='r', s=50, label='$2^{nd}\ Supernova$', zorder=4)
        ax_gal.scatter(df.iloc[-1]['x']*u.m.to(u.kpc), df.iloc[-1]['y']*u.m.to(u.kpc), marker='*', color='y', s=100, label='$BNS\ Merger$', zorder=4)
        xdir = (df.iloc[0]['vx']-dfi.iloc[0]['vx'])/1e5
        ydir = (df.iloc[0]['vy']-dfi.iloc[0]['vy'])/1e5
        ax_gal.arrow(df.iloc[0]['x']*u.m.to(u.kpc), df.iloc[0]['y']*u.m.to(u.kpc), dx=xdir, dy=ydir, color='r', width=0.02, head_width=0.1, zorder=4)

        # add colorbar
        cbar = fig.colorbar(pts, ticks=[age_col.min(), age_col.max()], cax=ax_cbar, orientation='vertical')

        # label & format
        ax_gal.set_xlim(-1*scale,scale)
        ax_gal.set_ylim(-1*scale,scale)
        ax_gal.set_xlabel(r'$kpc$')
        ax_gal.set_ylabel(r'$kpc$')
        cbar.set_label(r'$Gyr$')

        # annotate with apre, vkick, mhe
        ax_gal.annotate('$V_{kick} = %.1f\ km/s$\n$M_{He} = %.1f\ M_{\odot}$\n$a_{pre} = %.1f\ R_{\odot}$' % (dfi.iloc[0]['Vkick']*u.m.to(u.km),dfi.iloc[0]['Mhe']*u.kg.to(u.Msun),dfi.iloc[0]['Apre']*u.m.to(u.Rsun)), xy=(-4.8,3.5))

        ax_gal.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('traj_'+num+'.png', dpi=300)


    def trajectory_3d(self, traj_file):
        df = pd.read_csv(traj_file+'.dat')
        dfi = pd.read_csv(traj_file+'_ini.dat')

        # setup axes
        fig = plt.figure()
        gs = gridspec.GridSpec(8,9)
        ax_gal = fig.add_subplot(gs[:,:-1], projection='3d')
        ax_cbar = fig.add_subplot(gs[:,-1])

        # plot stellar background
        scale = 5.0
        def rho(r):
           return 2 * r * self.abulge/C.kpc.value * (self.abulge/C.kpc.value + r)**(-3)
        def norm(x):
            return (x-x.min())/(x.max()-x.min())

        x = np.linspace(-1*scale, scale, 50)
        y = np.linspace(-1*scale, scale, 50)
        z = np.linspace(-1*scale, scale, 50)
        xx, yy, zz = np.meshgrid(x,y,z)
        col = rho(np.sqrt(xx.flatten()**2 + yy.flatten()**2 + zz.flatten()**2))
        col_norm = norm(col)

        ax_gal.scatter(xx, yy, zz, c=col_norm, cmap='Greys', label=None, zorder=-1, alpha=0.3, s=0.1)


        # plot pre-SN circular orbit
        from scipy.integrate import ode
        Ri = np.sqrt(df.iloc[0]['x']**2+df.iloc[0]['y']**2+df.iloc[0]['z']**2)
        Vi = np.sqrt(dfi.iloc[0]['vx']**2+dfi.iloc[0]['vy']**2+dfi.iloc[0]['vz']**2)
        Torb = 2*np.pi*Ri/Vi
        solver=ode(self.gal.Vdot).set_integrator('dopri5', nsteps=1e13, max_step=u.year.to(u.s)*1e6, rtol=1e-11)
        sol = []
        def solout(t,y):
            temp=list(y)
            temp.append(t)
            sol.append(temp)
        RR = np.array([dfi.iloc[0]['vx'],dfi.iloc[0]['vy'],dfi.iloc[0]['vz'],df.iloc[0]['x'],df.iloc[0]['y'],df.iloc[0]['z']])
        solver.set_solout(solout)
        solver.set_initial_value(RR,0)
        solver.integrate(Torb)
        sol=np.array(sol)
        
        ax_gal.plot(sol[:,3]*u.m.to(u.kpc), sol[:,4]*u.m.to(u.kpc), sol[:,5]*u.m.to(u.kpc), color='g', label='initial orbit', zorder=2, alpha=1.0)

        # plot the trajectory of the binary through the galaxy, colored by age
        age_col = df['t']*u.s.to(u.Gyr)
        pts = ax_gal.scatter(df['x']*u.m.to(u.kpc), df['y']*u.m.to(u.kpc), df['z']*u.m.to(u.kpc), c=age_col, cmap='viridis', s=0.5, zorder=1, label=None, alpha=1.0)

        # plot location of supernova and kilonova
        ax_gal.scatter(df.iloc[0]['x']*u.m.to(u.kpc), df.iloc[0]['y']*u.m.to(u.kpc), df.iloc[0]['z']*u.m.to(u.kpc),marker='*', color='g', s=40, label='SN2', zorder=3)
        ax_gal.scatter(df.iloc[-1]['x']*u.m.to(u.kpc), df.iloc[-1]['y']*u.m.to(u.kpc), df.iloc[-1]['z']*u.m.to(u.kpc),marker='*', color='r', s=80, label='Merger', zorder=3)
        xdir = (df.iloc[0]['vx']-dfi.iloc[0]['vx'])/1e6
        ydir = (df.iloc[0]['vy']-dfi.iloc[0]['vy'])/1e6
        zdir = (df.iloc[0]['vz']-dfi.iloc[0]['vz'])/1e6
        ax_gal.quiver(df.iloc[0]['x']*u.m.to(u.kpc), df.iloc[0]['y']*u.m.to(u.kpc), df.iloc[0]['z']*u.m.to(u.kpc),xdir, ydir, zdir, color='r', length=10.0)

        # add colorbar
        cbar = fig.colorbar(pts, ticks=[age_col.min(), age_col.max()], cax=ax_cbar, orientation='vertical')

        # label & format
        ax_gal.set_xlim(-1*scale,scale)
        ax_gal.set_ylim(-1*scale,scale)
        ax_gal.set_zlim(-1*scale,scale)
        ax_gal.set_xlabel(r'$kpc$')
        ax_gal.set_ylabel(r'$kpc$')
        ax_gal.set_zlabel(r'$kpc$')
        cbar.set_label(r'$Gyr$')

        # annotate with apre, vkick, mhe
        #ax_gal.annotate('Vkick = %.1f km/s\nMhe = %.1f Msun\nApre = %.1f Rsun' % (dfi.iloc[0]['Vkick']*u.m.to(u.km),dfi.iloc[0]['Mhe']*u.kg.to(u.Msun),dfi.iloc[0]['Apre']*u.m.to(u.Rsun)), xy=(-4.8,3.5))

        ax_gal.legend()
        plt.tight_layout()
        plt.savefig('traj_3d.png', dpi=300)



    def trajectory_animation(self, traj_file):
        df = pd.read_csv(traj_file+'.dat')
        dfi = pd.read_csv(traj_file+'_ini.dat')
        num = traj_file[10:]

        # setup axes
        fig = plt.figure()
        gs = gridspec.GridSpec(8,9)
        ax_gal = fig.add_subplot(gs[:,:-1])
        ax_cbar = fig.add_subplot(gs[:,-1])

        # plot stellar background
        def norm(x):
            return (x-x.min())/(x.max()-x.min())
        col_norm = norm(self.col)
        scale = 5.0
        x = np.linspace(-1*scale, scale, 100)
        y = np.linspace(-1*scale, scale, 100)
        xx, yy = np.meshgrid(x,y)

        ax_gal.scatter(xx, yy, c=col_norm, cmap='Greys', label=None, zorder=1)

        # plot pre-SN circular orbit
        from scipy.integrate import ode
        Ri = np.sqrt(df.iloc[0]['x']**2+df.iloc[0]['y']**2+df.iloc[0]['z']**2)
        Vi = np.sqrt(dfi.iloc[0]['vx']**2+dfi.iloc[0]['vy']**2+dfi.iloc[0]['vz']**2)
        Torb = 2*np.pi*Ri/Vi
        solver=ode(self.gal.Vdot).set_integrator('dopri5', nsteps=1e13, max_step=u.year.to(u.s)*1e6, rtol=1e-11)
        sol = []
        def solout(t,y):
            temp=list(y)
            temp.append(t)
            sol.append(temp)
        RR = np.array([dfi.iloc[0]['vx'],dfi.iloc[0]['vy'],dfi.iloc[0]['vz'],df.iloc[0]['x'],df.iloc[0]['y'],df.iloc[0]['z']])
        solver.set_solout(solout)
        solver.set_initial_value(RR,0)
        solver.integrate(Torb)
        sol=np.array(sol)
        

        # label & format
        ax_gal.set_xlim(-1*scale,scale)
        ax_gal.set_ylim(-1*scale,scale)
        ax_gal.set_xlabel(r'$kpc$')
        ax_gal.set_ylabel(r'$kpc$')

        # annotate with apre, vkick, mhe and set legend before loop
        ax_gal.annotate('$V_{kick} = %.1f\ km/s$\n$M_{He} = %.1f\ M_{\odot}$\n$a_{pre} = %.1f\ R_{\odot}$' % (dfi.iloc[0]['Vkick']*u.m.to(u.km),dfi.iloc[0]['Mhe']*u.kg.to(u.Msun),dfi.iloc[0]['Apre']*u.m.to(u.Rsun)), xy=(-4.8,3.5))
        ax_gal.scatter(10, 10, marker='*', color='r', s=50, label='$2^{nd}\ Supernova$')
        ax_gal.arrow(10, 10, dx=1, dy=1, color='r', width=0.02, head_width=0.1)
        ax_gal.scatter(10, 10, marker='*', color='y', s=100, label='$BNS\ Merger$')
        ax_gal.legend(loc='upper right')

        xdir = (df.iloc[0]['vx']-dfi.iloc[0]['vx'])/1e5
        ydir = (df.iloc[0]['vy']-dfi.iloc[0]['vy'])/1e5

        age_col = df['t']*u.s.to(u.Gyr)
        pts = ax_gal.scatter(10*np.ones(len(age_col)),10*np.ones(len(age_col)), c=age_col, cmap='viridis')
        cbar = fig.colorbar(pts, ticks=[age_col.min(), age_col.max()], cax=ax_cbar, orientation='vertical')
        cbar.set_label(r'$Gyr$')
        plt.tight_layout()

        # create new df with the pertinent info
        before = np.transpose([sol[:,3][::-1],sol[:,4][::-1],sol[:,5][::-1],-1*sol[:,6][::-1]])
        after = np.transpose([df['x'],df['y'],df['z'],df['t']])
        df_all = pd.DataFrame(np.vstack([before[:-1],after]), columns=['x','y','z','t'])

        t_ctr = df_all['t'].min()
        res=len(df_all)/5
        stepsize = (df_all['t'].max()-df_all['t'].min())/res
        time = ax_gal.annotate('$t_{post_{SN}}=%.3f\ Gyr$' % (float(t_ctr)*u.s.to(u.Gyr)), xy=(-1.5,-4.7))
        print 'Number of steps: %i' % res
        for i in xrange(res):
            time.remove()
            time = ax_gal.annotate('$t_{post_{SN}}=%.3f$' % (float(t_ctr)*u.s.to(u.Gyr)), xy=(-1.5,-4.7))
            steps = df_all[(df_all['t']>t_ctr) & (df_all['t']<t_ctr+stepsize)]
            ax_gal.scatter(steps['x']*u.m.to(u.kpc), steps['y']*u.m.to(u.kpc), c=steps['t']*u.s.to(u.Gyr), vmin=0.0, vmax=age_col.max(), cmap='viridis', s=0.5, alpha=0.5, zorder=3)
            if t_ctr > 0.0:
                ax_gal.scatter(df_all.iloc[np.where(df_all['t']==0)]['x']*u.m.to(u.kpc), df_all.iloc[np.where(df_all['t']==0)]['y']*u.m.to(u.kpc), marker='*', color='r', s=50, zorder=4)
                ax_gal.arrow(float(df_all.iloc[np.where(df_all['t']==0)]['x']*u.m.to(u.kpc)), float(df_all.iloc[np.where(df_all['t']==0)]['y']*u.m.to(u.kpc)), dx=xdir, dy=ydir, color='r', width=0.02, head_width=0.1, zorder=4)
            if i == res-1:
                ax_gal.scatter(df_all.iloc[-1]['x']*u.m.to(u.kpc), df_all.iloc[-1]['y']*u.m.to(u.kpc), marker='*', color='y', s=100, zorder=4)
            plt.savefig('movie/' + num + '_' + str(i).zfill(4), dpi=200)
            t_ctr += stepsize
            print "...done with %i" % i

