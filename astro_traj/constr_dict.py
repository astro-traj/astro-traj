# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
#
# This file is part of gwemlightcurves.
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

"""`sample`
"""

from astropy.table import Table
import numpy as np

def GW(filename_samples):
    """
    Read in posterior samples and construct a dict of import params
    """

    GW = {}
    samples_out = Table.read(filename_samples, format='ascii')
    GW['m1'] = np.mean(samples_out['m1'])
    GW['m2'] = np.mean(samples_out['m2'])
    GW['m1_sigma'] = np.std(samples_out['m1'])
    GW['m2_sigma'] = np.std(samples_out['m2'])

    return GW


def galaxy(galaxy_name, filename_samples, r_eff, offset, h):

    # Dic of Galaxies containing dicts about properities
    Galaxy_Dict = {

        'NGC': {   
            'Mspiral': 0.0,                     # mass of the spiral (Msun) # NOTE: this information is not available, for now set to 0
            'Mbulge': (10**10.454)/h**2,        # Mstellar from 2MASS (Msun)
            'Mhalo': (10**12.2)/h,              # Mhalo from 2MASS (Msun)
            'D1': 0.81,                         # major axis from 2MASS (arcmin)
            'D2': 0.73,                         # minor axis from 2MASS (arcmin)
        }
    }

    # Dic of Galaxies containing dicts about properities
    samples_out = Table.read(filename_samples, format='ascii')

    Galaxy = Galaxy_Dict[galaxy_name]
    Galaxy['d'] = np.mean(samples_out['distance'])
    Galaxy['R_eff'] = r_eff
    Galaxy['offset'] = offset

    return Galaxy
    

def telescope(telescope_name):

    # Infer about the telescope that made the measurements (for angular resolution)
    telescope_dict = {
        'ESO' : {
            'D': 1.52,                          # diameter of telescope (m)
            'lambda': 650                       # wavelenth of light (nm)
        }
    }

    return telescope_dict[telescope_name]
