.. _examples:

########################################
Determining Charateristics of Progenitor
########################################


************
Introduction
************

The scenario this code attempts to address is the following. Say you observed a binary system in a galaxy that was a certain off-set from the center of that galaxy. You also have information about the masses of the system you observed. What, if anything, can you say about some of the properities of the system at its initial creation in the galaxy. Specifically, at the creation of the larger component mass of the system via a Supernova. The properities of the system at this stage that one needs to sample over inclue such as pre-supernova semi major axis, supernova kick, Mass of pre-supernova helium star, etc.

In order to accomplish this, one would need to load in properities of the galaxy in which this system was created. For example this includes information such as::

    'Mspiral' # mass of the spiral (Msun) # NOTE: this information is not available, for now set to 0
    'Mbulge' # Mstellar from 2MASS (Msun)
    'Mhalo'  # Mhalo from 2MASS (Msun)
    'D1':    # major axis from 2MASS (arcmin)
    'D2':    # minor axis from 2MASS (arcmin)

    >>> Galaxy = constr_dict.galaxy(galaxy_name, samples, r_eff, offset, h)

This is done using :meth:`~astro_traj.constr_dict.galaxy`. In addition to creating a dictionary of galaxy properities, we must assume a Galaxy model for this galaxy. The current model that is implemented and works is Hernquist :meth:`~astro_traj.galaxy.Hernquist_NFW`.::

    >>> gal=Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], h, rcut=100)

Once, the galaxy and galaxy model is selected it is now time to sample over a large set of initial conditions of the system that you observed. These initial conditions include, the mass of pre supernova helium star, the kick velocity of the  supernova, the pre supernova semi major axis, initial distance from center of galaxy. After creating these intiail conditions we evolve the system to see if a.) there is a merger within some reasonable time (<14 Giga years) and b.) it merged at the appropriate off-set (with some error) from the center of the galaxy. The only value sampled over in the following section that is not relevant to the initial conditions of the system that you observed is the distance uncertianity of the hose galaxy (which impacts the uncertainity in the observed off set). This value is not necessary as fixed inflated errors bars can be used in order to be "safe". In the following sections, we dive into the code used to accomplish this.

***************************
Sampling Initial Conditions
***************************

In the executable, all of the sampling is done as such::

    Nsys=args.trials
    dEfrac = 0.0

    # Initialize random draws for some parameters based on number of trials
    print "\nSampling binary parameters..."

    Mcomp_dist, Mns_dist = samp.sample_masses(samples, method=args.Ms, size=Nsys)
    # (Msun)

    d_dist = samp.sample_distance(samples, method=args.distance, size=Nsys)
    # (Mpc) (Not necessarily used, as inflated error bars in the observed galactic offset can be used to account for the distance (or uncertianity in distance) of the galaxy


    Apre_dist = samp.sample_Apre(Amin=0.1, Amax=10.0, method=args.Apre, size=Nsys)
    # (Rsun)

    epre_dist = samp.sample_epre(method=args.epre, size=Nsys)
    # (dimensionless)

    PDFR = samp.initialize_R()

    R_dist = samp.sample_R(PDFR, Nsys)
    # (kpc)

    PDFMhe = samp.initialize_Mhe(0.6)
    ECSPDFMhe = samp.initialize_Mhe(0.1)
    CCSPDFMhe = samp.initialize_Mhe(1.0)
    dumrand = np.random.uniform(0,1,size=Nsys)
    Mhe_dist = samp.sample_Mhe(Mmin=Mns_dist, method=args.Mhe, size=Nsys, PDF=PDFMhe, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand)
    # (Msun)
    ECS,CCS = samp.initialize_Vkick()

    Vkick_dist = samp.sample_Vkick(method=args.Vkick, size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=Mhe_dist, irand=dumrand)

Component and Secondary Mass
============================
:meth:`~astro_traj.sample.Sample.sample_masses`

The available methods for sampling are 'gaussian', 'mean', 'median', or 'posterior'::

The posterior method is used by default. This simply means that we draw samples directly from Gravitational Wave parameter estimation pdf of the source frame masses of the observed binary system::

    >>> Mcomp_dist, Mns_dist = samp.sample_masses(samples, method=args.Ms, size=Nsys)

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))

    >>> Mcomp_dist_posterior, Mns_dist_posterior = samp.sample_masses(samples, method='posterior', size=Nsys)
    >>> Mcomp_dist_median, Mns_dist_median = samp.sample_masses(samples, method='median', size=Nsys)
    >>> Mcomp_dist_mean, Mns_dist_mean = samp.sample_masses(samples, method='mean', size=Nsys)
    >>> Mcomp_dist_gaussian, Mns_dist_gaussian = samp.sample_masses(samples, method='gaussian', size=Nsys)
     

    >>> plot, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex=True, figsize=(18.5, 10.5))
    >>> ax1.hist(Mcomp_dist_posterior, bins=bins)
    >>> ax2.hist(Mcomp_dist_median, bins=bins)
    >>> ax3.hist(Mcomp_dist_mean, bins=bins)
    >>> ax4.hist(Mcomp_dist_gaussian, bins=bins)
    >>> ax5.hist(Mns_dist_posterior, bins=bins)
    >>> ax6.hist(Mns_dist_median, bins=bins)
    >>> ax7.hist(Mns_dist_mean, bins=bins)
    >>> ax8.hist(Mns_dist_gaussian, bins=bins)
    >>> ax1.set_xlabel('Comp Mass: Posterior', fontdict=font)
    >>> ax2.set_xlabel('Comp Mass: Median', fontdict=font)
    >>> ax3.set_xlabel('Comp Mass: Mean', fontdict=font)
    >>> ax4.set_xlabel('Comp Mass: Gaussian', fontdict=font)
    >>> ax5.set_xlabel('NS Mass: Posterior', fontdict=font)
    >>> ax6.set_xlabel('NS Mass: Median', fontdict=font)
    >>> ax7.set_xlabel('NS Mass: Mean', fontdict=font)
    >>> ax8.set_xlabel('NS Mass: Gaussian', fontdict=font)
    >>> plot.show()

Distance
========
:meth:`~astro_traj.sample.Sample.sample_distance`

The default method is median (i.e. the median value from the Gravitational Wave parameter estimation pdf of distance. Again, this value is critical in calculating the uncertainity in the observed offset and can be circumvented by smartly chosen inflated error bars in the observed of set::

    >>> d_dist = samp.sample_distance(samples, method=args.distance, size=Nsys) # (Mpc)

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))

    >>> d_dist_posterior = samp.sample_distance(samples, method='posterior', size=Nsys)
    >>> d_dist_median = samp.sample_distance(samples, method='median', size=Nsys)
    >>> d_dist_mean = samp.sample_distance(samples, method='mean', size=Nsys)
    >>> d_dist_gaussian = samp.sample_distance(samples, method='gaussian', size=Nsys)

    >>> plot, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(18.5, 10.5))
    >>> ax1.hist(d_dist_posterior, bins=bins)
    >>> ax2.hist(d_dist_median, bins=bins)
    >>> ax3.hist(d_dist_mean, bins=bins)
    >>> ax4.hist(d_dist_gaussian, bins=bins)
    >>> ax1.set_xlabel('Distance: Posterior', fontdict=font)
    >>> ax2.set_xlabel('Distance: Median', fontdict=font)
    >>> ax3.set_xlabel('Distance: Mean', fontdict=font)
    >>> ax4.set_xlabel('Distance: Gaussian', fontdict=font)
    >>> plot.show()

Pre Supernova Semi Major Axis
=============================
:meth:`~astro_traj.sample.Sample.sample_Apre`

The available methods for sampling are 'uniform' and 'log'. This value is the pre-supernova semi major axis.::

    >>> Apre_dist = samp.sample_Apre(Amin=0.1, Amax=10.0, method='uniform', size=Nsys)

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))

    >>> Apre_dist_log = samp.sample_Apre(Amin=0.1, Amax=10.0, method='log', size=Nsys)
    >>> Apre_dist_uniform = samp.sample_Apre(Amin=0.1, Amax=10.0, method='uniform', size=Nsys)

    >>> plot, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(18.5, 10.5))
    >>> ax1.hist(Apre_dist_log, bins=bins)
    >>> ax2.hist(Apre_dist_uniform, bins=bins)
    >>> ax1.set_xlabel('Apre: Log', fontdict=font)
    >>> ax2.set_xlabel('Apre: Uniform', fontdict=font)
    >>> plot.show()

Pre Supernova eccentricity
==========================
Because we assume circular orbits, we assume that the eccentricity of the system pre second supernova is neglible (set to 0 here), but do account for effects of eccentricity post second supernova.
:meth:`~astro_traj.sample.Sample.sample_epre`

The available method for sampling is 'circularized'::

    >>> epre_dist = samp.sample_epre(method='circularized', size=Nsys)


.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))

    >>> epre_dist_circularized = samp.sample_epre(method='circularized', size=Nsys)

    >>> plot, ax1 = plt.subplots(1, sharex=True)
    >>> ax1.hist(epre_dist_circularized, bins=bins)
    >>> ax1.set_xlabel('Eccentricity Pre-Supernova: Circularized', fontdict=font)
    >>> plot.show()


Initialize Off Set From Galactic Center
=======================================
We create a custom distribution to sample the intiial galactic offset of the pre second supernova system. The r_eff is used to control sampling at galactic offsets that are unrealistically far away from the center of the galaxy (i.e. we expect less binaries to form super far away from the galactic center and moreover some initial offsets may in fact be outside the plausible "diameter" of the galaxy. To initialize this PDF, we use :meth:`~astro_traj.sample.Sample.initialize_R`, to sample some number of outcomes from this PDF we use :meth:`~astro_traj.sample.Sample.sample_R`::

    >>> PDFR = samp.initialize_R()
    >>> R_dist = samp.sample_R(PDFR, Nsys) # (kpc)

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))

    >>> PDFR = samp.initialize_R()
    >>> R_dist = samp.sample_R(PDFR, Nsys)

    >>> plot, ax1 = plt.subplots(1, sharex=True)
    >>> ax1.hist(R_dist, bins=bins)
    >>> ax1.set_xlabel('Initial Off Set from Galactic Center', fontdict=font)
    >>> plot.show()

Mass of Pre Supernova Helium Star
=================================
:meth:`~astro_traj.sample.Sample.initialize_Mhe`
:meth:`~astro_traj.sample.Sample.sample_Mhe`::

Available methods include 'power', 'uniform', 'beniamini2'.

`Beniamini2 <https://arxiv.org/pdf/1510.03111.pdf#equation.4.7>`_ draws from two distributions, low eccentricity (ECS) and high eccentricity (CCSN), for the pre-supernova Helium Star and Kick Velocity distributions. It does so in a  60 40 split which is motivated by the number of such systems we observe in the Milky Way (6 and 4) which is shown in `Figure 2 <https://arxiv.org/pdf/1510.03111.pdf#figure.2>`_. The initialized values for the distribution are motivated form the paper where the deltaM_0 and Vkick_0 for ECS that corresponded to the maximum likelihood weere 0.1 and 5.0, respectively, and the deltaM_0 and Vkick_0 for ECS that corresponded to the maximum likelihood for CCSN were 1.0 and 158.0 respectively.::

    >>> ECSPDFMhe = samp.initialize_Mhe(0.1)
    >>> CCSPDFMhe = samp.initialize_Mhe(1.0)
    >>> dumrand = np.random.uniform(0,1,size=Nsys)
    >>> Mhe_dist = samp.sample_Mhe(Mmin=Mns_dist, method='uniform', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))
    >>> Mcomp_dist, Mns_dist = samp.sample_masses(samples, method='posterior', size=Nsys)

    >>> ECSPDFMhe = samp.initialize_Mhe(0.1)
    >>> CCSPDFMhe = samp.initialize_Mhe(1.0)
    >>> dumrand = np.random.uniform(0,1,size=Nsys)
    >>> Mhe_dist_uniform = samp.sample_Mhe(Mmin=Mns_dist, method='uniform', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)
    >>> Mhe_dist_power = samp.sample_Mhe(Mmin=Mns_dist, method='power', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)
    >>> Mhe_dist_beniamini2 = samp.sample_Mhe(Mmin=Mns_dist, method='beniamini2', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)

    >>> plot, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(18.5, 10.5))
    >>> ax1.hist(Mhe_dist_uniform, bins=bins)
    >>> ax2.hist(Mhe_dist_power, bins=bins)
    >>> ax3.hist(Mhe_dist_beniamini2, bins=bins)
    >>> ax1.set_xlabel('Mass Helium Star: Uniform', fontdict=font)
    >>> ax2.set_xlabel('Mass Helium Star: Power', fontdict=font)
    >>> ax3.set_xlabel('Mass Helium Star: Beniamini2', fontdict=font)
    >>> plot.show()


Supernova Kick Velocity
=======================
:meth:`~astro_traj.sample.Sample.initialize_Vkick`
:meth:`~astro_traj.sample.Sample.sample_Vkick`

Available methods include 'maxwellian', 'uniform', 'beniamini2'.

`Beniamini2 <https://arxiv.org/pdf/1510.03111.pdf#equation.4.7>`_ draws from two distributions, low eccentricity (ECS) and high eccentricity (CCSN), for the pre-supernova Helium Star and Kick Velocity distributions. It does so in a  60 40 split which is motivated by the number of such systems we observe in the Milky Way (6 and 4) which is shown in `Figure 2 <https://arxiv.org/pdf/1510.03111.pdf#figure.2>`_. The initialized values for the distribution are motivated form the paper where the deltaM_0 and Vkick_0 for ECS that corresponded to the maximum likelihood weere 0.1 and 5.0, respectively, and the deltaM_0 and Vkick_0 for ECS that corresponded to the maximum likelihood for CCSN were 1.0 and 158.0 respectively.::

    >>> ECS,CCS = samp.initialize_Vkick()
    >>> Vkick_dist = samp.sample_Vkick(method=args.Vkick, size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 22}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))
    >>> dumrand = np.random.uniform(0,1,size=Nsys)

    >>> ECS,CCS = samp.initialize_Vkick()
    >>> Vkick_dist_uniform = samp.sample_Vkick(method='uniform', size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)
    >>> Vkick_dist_maxwellian = samp.sample_Vkick(method='maxwellian', size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)
    >>> Vkick_dist_beniamini2 = samp.sample_Vkick(method='beniamini2', size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)

    >>> plot, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(18.5, 10.5))
    >>> ax1.hist(Vkick_dist_uniform, bins=bins)
    >>> ax2.hist(Vkick_dist_maxwellian, bins=bins)
    >>> ax3.hist(Vkick_dist_beniamini2, bins=bins)
    >>> ax1.set_xlabel('Supernova Kick Velocity: Uniform', fontdict=font)
    >>> ax2.set_xlabel('Supernova Kick Velocity: Maxwellian', fontdict=font)
    >>> ax3.set_xlabel('Supernova Kick Velocity: Beniamini2', fontdict=font)
    >>> plot.show()

********************************
Creating and Evolving the System
********************************
Now that we have successfully sampled a number of initial conditions for our pre-Supernova binary, we can create the :class:`~astro_traj.system.System`::

    >>> # initialize System class with pertinent parameters
    >>> T=System(gal, R, Mns, Mcomp, Mhe, Apre, epre, d, Vkick, sys_flag=args.sys_flag)

In the following sub-sections we discuss the forward modelling that occurs after the creation of this system.

Supernova
=========
The very first thing that you have to do is take the initial conditions of your pre-Supernova Helium star + neutron star system, and evolve Helium star through supernova to the post supernova object + neutron star system. Naturally, not all initial conditions of pre-supernova helium star semi-major axis, etc. will successfully result in the creation of a binary system. Therefore, the first check of the forward modeling occurs, that is, does a successful binary result from the supernova.

System Resulting from Supernova of Helium Star
----------------------------------------------
:meth:`~astro_traj.system.System.SN`

We utilize `Kalogera 1996 <http://iopscience.iop.org/article/10.1086/177974/meta>`_ in order to determine the properities of the post supernova binary (such as eccentricity of the system post supernova, additional mass etc.) From the documnetation, We use Eq 1, 3, 4, and 34: giving Vr, Apost, epost, and (Vsx,Vsy,Vsz), respectively. Also see Fig 1 in that paper for coordinate system. After calculating this post super system values we check for whether this system would be expected to result in the successful creation of a binary. We check 4 separate equations detailed in `Willems et al 2002 <http://iopscience.iop.org/article/10.1086/429557/meta>`_. Specifically, we use eq 21, 22, 23, 24, 25, 26 for checks of SN survival::

    >>> T.SN()

.. plot::

    >>> from astro_traj.galaxy import Hernquist_NFW
    >>> from astro_traj import constr_dict
    >>> from astro_traj.sample import Sample
    >>> from astro_traj.system import System
    >>> import numpy as np
    >>> from matplotlib import use
    >>> use('agg')
    >>> import matplotlib.pyplot as plt
    >>> font = {'size': 16}

    >>> samples = 'posterior_samples.dat'
    >>> Galaxy = constr_dict.galaxy('NGC', samples, 2.8, 5, 0.679)
    >>> gal = Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], 0.679, rcut=100)
    >>> samp = Sample(gal)
    >>> Nsys = 1000
    >>> bins = int(np.round(np.sqrt(Nsys)))
    >>> dumrand = np.random.uniform(0,1,size=Nsys)


    >>> Mcomp_dist, Mns_dist = samp.sample_masses(samples, method='posterior', size=Nsys)
    >>> d_dist = samp.sample_distance(samples, method='median', size=Nsys)
    >>> Apre_dist_log = samp.sample_Apre(Amin=0.1, Amax=10.0, method='log', size=Nsys)
    >>> Apre_dist_uniform = samp.sample_Apre(Amin=0.1, Amax=10.0, method='uniform', size=Nsys)
    >>> epre_dist = samp.sample_epre(method='circularized', size=Nsys)
    >>> PDFR = samp.initialize_R()
    >>> R_dist = samp.sample_R(PDFR, Nsys)
    >>> ECSPDFMhe = samp.initialize_Mhe(0.1)
    >>> CCSPDFMhe = samp.initialize_Mhe(1.0)
    >>> Mhe_dist_uniform = samp.sample_Mhe(Mmin=Mns_dist, method='uniform', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)
    >>> Mhe_dist_power = samp.sample_Mhe(Mmin=Mns_dist, method='power', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)
    >>> Mhe_dist_beniamini2 = samp.sample_Mhe(Mmin=Mns_dist, method='beniamini2', size=Nsys, PDF=None, ECSPDF=ECSPDFMhe, CCSPDF=CCSPDFMhe, irand=dumrand) # (Msun)
    >>> ECS, CCS = samp.initialize_Vkick()
    >>> Vkick_dist_uniform = samp.sample_Vkick(method='uniform', size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)
    >>> Vkick_dist_maxwellian = samp.sample_Vkick(method='maxwellian', size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)
    >>> Vkick_dist_beniamini2 = samp.sample_Vkick(method='beniamini2', size=Nsys, ECSPDF=ECS, CCSPDF=CCS, Mhe=None, irand=dumrand)

    >>> successful_binaries_log_ben2_ben2 = []
    >>> successful_binaries_uni_uni_uni = []
    >>> successful_binaries_uni_power_maxwell = []

    >>> for R, d, Mcomp, Mns, Apre, epre, Mhe, Vkick in zip(R_dist, d_dist, Mcomp_dist, Mns_dist, Apre_dist_log, epre_dist, Mhe_dist_beniamini2, Vkick_dist_beniamini2):
    >>>     T = System(gal, R, Mns, Mcomp, Mhe, Apre, epre, d, Vkick, sys_flag=None)
    >>>     T.SN()
    >>>     if T.flag != 3:
    >>>         successful_binaries_log_ben2_ben2.append(True)
    >>>     else:
    >>>         successful_binaries_log_ben2_ben2.append(False)

    >>> for R, d, Mcomp, Mns, Apre, epre, Mhe, Vkick in zip(R_dist, d_dist, Mcomp_dist, Mns_dist, Apre_dist_uniform, epre_dist, Mhe_dist_uniform, Vkick_dist_uniform):
    >>>     T = System(gal, R, Mns, Mcomp, Mhe, Apre, epre, d, Vkick, sys_flag=None)
    >>>     T.SN()
    >>>     if T.flag != 3:
    >>>         successful_binaries_uni_uni_uni.append(True)
    >>>     else:
    >>>         successful_binaries_uni_uni_uni.append(False)


    >>> for R, d, Mcomp, Mns, Apre, epre, Mhe, Vkick in zip(R_dist, d_dist, Mcomp_dist, Mns_dist, Apre_dist_uniform, epre_dist, Mhe_dist_power, Vkick_dist_maxwellian):
    >>>     T = System(gal, R, Mns, Mcomp, Mhe, Apre, epre, d, Vkick, sys_flag=None)
    >>>     T.SN()
    >>>     if T.flag != 3:
    >>>         successful_binaries_uni_power_maxwell.append(True)
    >>>     else:
    >>>         successful_binaries_uni_power_maxwell.append(False)
    
    >>> plot, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3, sharex='col', figsize=(18.5, 10.5))
    >>> ax1.hist(Mhe_dist_beniamini2, bins=bins)
    >>> ax2.hist(Mhe_dist_uniform, bins=bins)
    >>> ax3.hist(Mhe_dist_power, bins=bins)
    >>> ax1.hist(Mhe_dist_beniamini2[successful_binaries_log_ben2_ben2], bins=bins)
    >>> ax2.hist(Mhe_dist_uniform[successful_binaries_uni_uni_uni], bins=bins)
    >>> ax3.hist(Mhe_dist_power[successful_binaries_uni_power_maxwell], bins=bins)
    >>> ax1.set_xlabel('Mhe: Apre: Log Mhe: ben2 Vkick: ben2', fontdict=font)
    >>> ax2.set_xlabel('Mhe: Apre: uni Mhe: uni Vkick: uni', fontdict=font)
    >>> ax3.set_xlabel('Mhe: Apre: uni Mhe: power Vkick: maxwell', fontdict=font)
  
    >>> ax4.hist(Vkick_dist_beniamini2, bins=bins)
    >>> ax5.hist(Vkick_dist_uniform, bins=bins)
    >>> ax6.hist(Vkick_dist_maxwellian, bins=bins)
    >>> ax4.hist(Vkick_dist_beniamini2[successful_binaries_log_ben2_ben2], bins=bins)
    >>> ax5.hist(Vkick_dist_uniform[successful_binaries_uni_uni_uni], bins=bins)
    >>> ax6.hist(Vkick_dist_maxwellian[successful_binaries_uni_power_maxwell], bins=bins)
    >>> ax4.set_xlabel('Vkick: Apre: Log Mhe: ben2 Vkick: ben2', fontdict=font)
    >>> ax5.set_xlabel('Vkick: Apre: uni Mhe: uni Vkick: uni', fontdict=font)
    >>> ax6.set_xlabel('Vkick: Apre: uni Mhe: power Vkick: maxwell', fontdict=font)
  
    >>> ax7.hist(Apre_dist_log, bins=bins)
    >>> ax8.hist(Apre_dist_uniform, bins=bins)
    >>> ax9.hist(Apre_dist_uniform, bins=bins)
    >>> ax7.hist(Apre_dist_log[successful_binaries_log_ben2_ben2], bins=bins)
    >>> ax8.hist(Apre_dist_uniform[successful_binaries_uni_uni_uni], bins=bins)
    >>> ax9.hist(Apre_dist_uniform[successful_binaries_uni_power_maxwell], bins=bins)
    >>> ax7.set_xlabel('Apre: Apre: Log Mhe: ben2 Vkick: ben2', fontdict=font)
    >>> ax8.set_xlabel('Apre: Apre: uni Mhe: uni Vkick: uni', fontdict=font)
    >>> ax9.set_xlabel('Apre: Apre: uni Mhe: power Vkick: maxwell', fontdict=font)
    >>> plot.suptitle('Number of Succesful Binaries Post Supernova for a Variety of Init Cond')
    >>> plot.show()


Evolving System Through Galaxy
==============================
In order to forward model successfully created binaries through the galaxy, one must calculate first how long it will take for the merger of the system in order to set the upper bound on the ODE used to evolve the Velocity (and X,Y,Z coordinates) of the system through the galaxy.

Time to Merger Peters 1964
--------------------------
:meth:`~astro_traj.system.System.setTmerge`
Checked against `Peters 1964 <https://doi.org/10.1103/PhysRev.136.B1224>`_::

    >>> # set merger time for trajectory integration, specify Tmin and Tmax in Gyr
    >>> T.setTmerge(Tmin=0.0, Tmax=14.0)

Distance of Binary From Center
------------------------------
First, you randomly select from initial XYZ direction.
:meth:`~astro_traj.system.System.setXYZ_0`
Next, you randomly select an initial velocity direction :meth:`~astro_traj.system.System.setVxyz_0`:
Then based on Tmerge you solve an ODE and evolve XYZ until merger.
:meth:`~astro_traj.system.System.doMotion`::

    >>> # choose random location on sphere of radius R
    >>> T.setXYZ_0()
    >>> # choose random direction for circular velocity, and add SN-imparted velocity to get V0
    >>> T.setVxyz_0()

    >>> # integrate trajectory until Tmerge
    >>> T.doMotion()


Check that conservation of energy is obeyed
-------------------------------------------
:meth:`~astro_traj.system.System.energy_check`
Calculate the initial Energy of the system and the final energy and make sure it is conserved to within some small error (0.001)::

    >>> # check for energy conservation, and hold onto highest offset
    >>> T.energy_check()
    >>> if T.dEfrac > dEfrac:
    >>>     dEfrac = T.dEfrac

Checking Offset
===============
Finally, You have a location for where in the galaxy your system merged. It is in XYZ so project onto XY plane and check if it matches to the observed off set (with some uncertainty in that offset accounted for).
:meth:`~astro_traj.system.System.check_success`


**********************************************
Exploring Seemingly Optimal SN Kick Velocities
**********************************************

Two special case runs include setting --system-flag = 'tangential' or --system-flag = 'radial_simple'.

These special realizations of the code ask the following questions: "Let us imagine the most seemingly optimal kick velocity direction of the SN that would result in getting us from some initial galactic offset that is smaller than the observed offset to the observed offset. This is the "radial_simple" case and it means that all of the kick velocity is "straight out" in the x direction. Conversely, let us imagine the most seemingly unoptimal way to get from some initial galactic offset that is smaller than the observed offset to the observed offset. This is the tangential case, which means all of velocity is perpendicular to the orbit." How would one explore the results from such systems?

We simply do a special toy sampling in kick velocity (uniform 0 to 1000), and initial offset from center of the galaxy (uniform from 0.001 to observed offset). [Code](https://github.com/astro-traj/astro-traj/blob/master/bin/LIGOTraj#L151)::

    if args.sys_flag=='radial_simple' or args.sys_flag=='tangential' or args.sys_flag=='radial_simple2' or args.sys_flag=='tangential2':

        R_dist = np.linspace(0.001,Galaxy['offset'],10)
        Vkick_dist=np.linspace(0,1000,100)
        RV=np.array([[rr,vv] for rr in R_dist for vv in Vkick_dist]).transpose()
        R_dist=RV[0]
        Vkick_dist=RV[1]

In addition, we check this toy sampling for a discrete set of realization of Mhs, Mhe, Mcomp and Apre. [Code](https://github.com/astro-traj/astro-traj/blob/master/bin/LIGOTraj#L168)::

    if args.sys_flag=='tangential' or args.sys_flag=='radial_simple':
        Mns = 1.097
        Mcomp = 1.713
        Mhe = 4.5
        Apre = 2.0

Assuming these types of kick directions will impact what initial Vsys, X0, Y0, Z0, ad Vx0, Vy0, and vz0 are calculated/assumed. For radial X0, is initial offset and vX0 is Vsys with Vy0 = vz0 = Y0 = z0 = 0. [Code](https://github.com/astro-traj/astro-traj/blob/master/bin/LIGOTraj#L202)::

    if T.sys_flag=='radial_simple' or T.sys_flag=='radial_simple2':
        if R>Galaxy['offset']:continue
        T.SN()
        T.flag=9
        T.X0,T.Y0,T.Z0=T.R,0.,0.
        T.Vx0,T.Vy0,T.Vz0 = T.V_sys,0.,0.
        T.Tmerge = 0.1*u.Gyr.to(u.s)
        T.doMotion()

For tangential, X.Y.Z and Vz,Vy,Vz are calculated as normal [Code](https://github.com/astro-traj/astro-traj/blob/master/bin/LIGOTraj#L182)::

    if T.sys_flag=='tangential' or T.sys_flag=='tangential2':
        if R>Galaxy['offset']:continue
        T.SN()
        T.setXYZ_0()
        T.setVxyz_0()
        T.flag=9
        T.Tmerge = 0.1*u.Gyr.to(u.s)
        T.doMotion()

but Vsys is [Code](https://github.com/astro-traj/astro-traj/blob/master/astro_traj/system.py#L299)::

    #Rotate by omega while keeping perpendicular to R
    Vp_rot = (Vp*np.cos(omega)) + (np.cross(k,Vp)*np.sin(omega))
    Vp_rot_tot = np.sqrt((Vp[0]**2)+(Vp[1]**2)+(Vp[2]**2))
    if self.sys_flag == 'tangential' or self.sys_flag== 'tangential2':
        vsys = [V_sys*Vp_rot[0]/Vp_rot_tot,V_sys*Vp_rot[1]/Vp_rot_tot,V_sys*Vp_rot[2]/Vp_rot_tot]

And in the SN kick calculated here  the kick vector is set as follows:

[Code](https://github.com/astro-traj/astro-traj/blob/master/astro_traj/system.py#L129)::

    if self.sys_flag == 'radial_simple' or self.sys_flag == 'tangential' or self.sys_flag == 'radial_simple2' or self.sys_flag == 'tangential2':
        Vkx,Vky,Vkz=0,-Vkick,0


*****************
Backward Modeling
*****************

Backward modeling would be to say let us strt with a system of known quanity (i.e. observed offset and some T merge) and see if we can work backwards and retrieve our initial conditions.
The assumptions made in the type of systems and the modeling of the velocity of the binary (F=ma), are simple enough that checks in the forward modeling such as conservation of energy are enough to ensure the sanity of the code.
