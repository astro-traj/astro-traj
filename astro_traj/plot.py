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

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as C
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.mlab import griddata
import corner
import pdb

__author__ = 'Michael Zevin <michael.zevin@ligo.org>'
__all__ = ['Plotting']

class Plot:
    def __init__(self, gal=None, output=None):
        '''
        initialize with values passed to gal and output file. outfile must be a string.
        '''
        if gal:
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
