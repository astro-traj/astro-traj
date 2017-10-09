.. _examples:

########################################
Determining Charateristics of Progenitor
########################################


************
Introduction
************

The scenario this code attempts to address is the following. Say you observed a binary system in a galaxy that was a certain off-set from the center of that galaxy. You also have information about the masses of the system you observed. What, if anything, can you say about some of the properities of the system, such as pre-supernova semi major axis, supernova kick, etc, that create that system.

In order to accomplish this, one would need to load in properities of the galaxy in which this system was created. For example this includes information such as::

    'Mspiral' # mass of the spiral (Msun) # NOTE: this information is not available, for now set to 0
    'Mbulge' # Mstellar from 2MASS (Msun)
    'Mhalo'  # Mhalo from 2MASS (Msun)
    'D1':    # major axis from 2MASS (arcmin)
    'D2':    # minor axis from 2MASS (arcmin)

    >>> Galaxy = constr_dict.galaxy(galaxy_name, samples, r_eff, offset, h)

This is done using :meth:`~astro_traj.constr_dict.galaxy`. In addition to creating a dictionary of galaxy properities, we must assume a Galaxy model for this galaxy. The current model that is implemented and works is Hernquist :meth:`~astro_traj.galaxy.Hernquist_NFW`.::

    >>> gal=Hernquist_NFW(Galaxy['Mspiral'], Galaxy['Mbulge'], Galaxy['Mhalo'], Galaxy['R_eff'], h, rcut=100)

Once, the galaxy and galaxy model is selected it is now time to sample over a large set of initial conditions of the system that you observed. These initial conditions include, the mass of pre supernova helium star, the kick velocity of the  supernova, the pre supernova semi major axis, initial distance from center of galaxy. After creating these intiail conditions we evolve the system to see if a.) there is a merger within some reasonable time (<14 Giga years) and b.) it merged at the appropriate off-set (with some error) from the center of the galaxy. In the following sections, we dive into the code used to accomplish this.

*********
Technique
*********

Sampling
========

In the executable, all of the sampling is done as such::

    Nsys=args.trials
    dEfrac = 0.0

    # Initialize random draws for some parameters based on number of trials
    print "\nSampling binary parameters..."

    Mcomp_dist, Mns_dist = samp.sample_masses(samples, method=args.Ms, size=Nsys)
    # (Msun)

    d_dist = samp.sample_distance(samples, method=args.distance, size=Nsys)
    # (Mpc)


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
----------------------------
:meth:`~astro_traj.sample.Sample.sample_masses`

The available methods for sampling are 'gaussian', 'mean', 'median', or 'posterior'::

The posterior method is used by default. This simply means that we draw samples directly from Gravitational Wave parameter estimation pdf of the source frame masses of the post supernova binary system::

    >>> Mcomp_dist, Mns_dist = samp.sample_masses(samples, method=args.Ms, size=Nsys)

.. plot::
   :include-source:

Distance
--------
:meth:`~astro_traj.sample.Sample.sample_distance`

The default method is median (i.e. the median value from the Gravitational Wave parameter estimation pdf of distance::

    >>> d_dist = samp.sample_distance(samples, method=args.distance, size=Nsys) # (Mpc)

.. plot::
   :include-source:

Pre Supernova Semi Major Axis
-----------------------------
:meth:`~astro_traj.sample.Sample.sample_Apre`

The available methods for sampling are 'gaussian', 'mean', 'median', or 'posterior'::

    >>> 

.. plot::
   :include-source:

Pre Supernova eccentricity
--------------------------
:meth:`~astro_traj.sample.Sample.sample_epre`


.. plot::
   :include-source:

Initialize Off Set From Center
------------------------------
:meth:`~astro_traj.sample.Sample.initialize_R`
:meth:`~astro_traj.sample.Sample.sample_R`

.. plot::
   :include-source:

Mass of Pre Supernova Helium Star
---------------------------------
:meth:`~astro_traj.sample.Sample.initialize_Mhe`
:meth:`~astro_traj.sample.Sample.sample_Mhe`

.. plot::
   :include-source:

Supernova Kick Velocity
-----------------------
:meth:`~astro_traj.sample.Sample.initialize_Vkick`
:meth:`~astro_traj.sample.Sample.sample_Vkick`

.. plot::
   :include-source:

Supernova
=========
The very first thing that you have to do is taken the initial conditions of your pre-Supernova Helium star + neutron star system, and evolve Helium star through supernova to the post supernova mass. After doing this, the first check of the system occurs. That is, does a successful component mass result from the supernova 

System Resulting from Supernova of Helium Star
----------------------------------------------
:meth:`~astro_traj.system.System.SN`
We utilize `Kalogera 1996 <http://iopscience.iop.org/article/10.1086/177974/meta>`_ From the documnetation, We use Eq 1, 3, 4, and 34: giving Vr, Apost, epost, and (Vsx,Vsy,Vsz) respectively Also see Fig 1 in that paper for coordinate system.

Checking Supernova Flags
------------------------
`Willems et al 2002 <http://iopscience.iop.org/article/10.1086/429557/meta>`_: We use eq 21, 22, 23, 24, 25, 26 for checks of SN survival

Evolving System
===============

Time to Merger Peters 1964
--------------------------
:meth:`~astro_traj.system.System.setTmerge`
Checked against `Peters 1964 <https://doi.org/10.1103/PhysRev.136.B1224>`_

Distance of Binary From Center
------------------------------
First, you randomly select from initial XYZ direction.
:meth:`~astro_traj.system.System.setXYZ_0`
Then based on Tmerge you solve an ODE and evolve XYZ until merger.
:meth:`~astro_traj.system.System.doMotion`

Checking Offset
===============
:meth:`~astro_traj.system.System.check_success`

