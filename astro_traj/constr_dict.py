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

from astropy.table import Table
from astro_traj.HaloMassClass import relation
import astropy.units as units
import astropy.constants as constants
import numpy as np

__author__ = ['Chase Kimball <charles.kimball@ligo.org>', 'Michael Zevin <michael.zevin@ligo.org>']
__credits__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

def NS_masses(samples=None):
    """
    Read in posterior samples and construct a dict of import params, or draw fiducial mass values
    """

    NS = {}
    gal_m1_mean = 1.33
    gal_m1_sigma = 0.09
    gal_m2_mean = 1.33
    gal_m2_sigma = 0.09
    if samples:
        samples_out = Table.read(samples, format='ascii')
        NS['m1'] = np.mean(samples_out['m1_source'])
        NS['m2'] = np.mean(samples_out['m2_source'])
        NS['m1_sigma'] = np.std(samples_out['m1_source'])
        NS['m2_sigma'] = np.std(samples_out['m2_source'])
    else:
        NS['m1'] = gal_m1_mean
        NS['m2'] = gal_m2_mean
        NS['m1_sigma'] = gal_m1_sigma
        NS['m2_sigma'] = gal_m2_sigma

    return NS


def galaxy(r_eff, h, stellar_mass=None, redshift=None, distance=None, galaxy_name=None):
    """
    Construct Galaxy dict
    Can either provide galaxy_name, which holds specific properties about a single galaxy, 
    or Mstellar and z/distance, from which we can use stellar mass/DM mass relation to construct dict
    """

    # First, make sure we have the necessary information
    if not redshift and not distance:
        raise ValueError("Must provide either redshift of luminosity distance!")
    if not stellar_mass and not galaxy_name:
        raise ValueError("Must provide either stellar mass or name of galaxy!")
    
    # If redshift if provided, convert to distance
    if redshift and not distance:
        c = constants.c.value * units.m.to(units.km) 
        v_recc = c*((redshift+1)**2 - 1)/((redshift+1)**2+1)   # km/s
        distance = v_recc / (100*h)   # Mpc

    # Dict of Galaxies containing dicts about properities
    Galaxy_Dict = {

        'NGC4993': {
            'Mspiral': 0.0,                     # mass of the spiral (Msun) # NOTE: this information is not available, for now set to 0
            'Mbulge': (10**10.454)/h**2,        # Mstellar from 2MASS (Msun)
            'Mhalo': (10**12.2)/h,              # Mhalo from 2MASS (Msun)
            'D1': 0.81,                         # major axis from 2MASS (arcmin)
            'D2': 0.73,                         # minor axis from 2MASS (arcmin)
            'redshift': 0.0,
            'R_eff': r_eff,
            'distance': distance
        }
    }

    # Dict containing properities of the host galaxy
    if galaxy_name:
        Galaxy = Galaxy_Dict[galaxy_name]
    else:
        Galaxy={}
        Galaxy['Mspiral'] = 0.0
        Galaxy['Mbulge'] = 10**stellar_mass
        Galaxy['redshift'] = redshift
        Galaxy['distance'] = distance
        Galaxy['R_eff'] = r_eff
        mstar_mhalo = relation(version='new')
        Galaxy['Mhalo'] = mstar_mhalo.getMhalo(stellar_mass, redshift)
        # FIXME: this doesn't seem right, check with Chase
        Galaxy['Mhalo'] = 10*Galaxy['Mbulge']

    return Galaxy


def offset(offset, distance, offset_uncer=None, telescope_name=None):
    """
    Store offset and offset uncertainty in a dict. Can either use user inputs or
    apporximate the uncertainty using telescope dict
    NOTE: If telescope is specified, then the offset and offset uncertainty must be provided
    in arcseconds. Otherwise, they should be provided in kpc. 
    """

    # First, make sure we have the necessary information
    if not offset_uncer and not telescope_name:
        raise ValueError("Must provide either offset uncertainty in kpc or name of telescope from telescope_dict!")
    if offset_uncer and telescope_name:
        raise ValueError("Either specify the offset and offset uncertainty in kpc, or the offset in arcseconds and the name of the telescope you wish to use to calculate the offset uncertainty!")

    # Infer about the telescope that made the measurements (for angular resolution)
    telescope_dict = {
        'ESO' : {
            'D': 1.52,                          # diameter of telescope (m)
            'lambda': 650                       # wavelenth of light (nm)
        }
    }

    # Dict containing properities of the host galaxy
    Offset={}
    if telescope_name:
        telescope = telescope_dict[telescope_name]
        offset = np.tan(206265*offset)*distance*units.Mpc.to(units.kpc)   # kpc
        Offset['offset'] = offset
        offset_uncer = distance*units.Mpc.to(units.kpc)*np.tan(1.22*tele['lambda']*units.nm.to(units.m) / tele['D'])
        Offset['offset_uncer'] = offset_uncer 
    else:
        Offset['offset'] = offset
        Offset['offset_uncer'] = offset_uncer

    return Offset
    

