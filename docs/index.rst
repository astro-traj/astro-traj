.. astro_traj documentation master file, created by
   sphinx-quickstart on Thu Apr 21 14:05:08 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to astro_traj's documentation!
======================================


Installing astro_traj
---------------------

The easiest method to install astro_traj is using `pip <https://pip.pypa.io/en/stable/>`_ directly from the `GitHub repository <https://github.com/astro-traj/astro-traj>`_:

.. code-block:: bash

   $ pip install git+https://github.com/astro-traj/astro-traj


Table of Contents
-----------------

.. toctree::
   :maxdepth: 4

   examples/index


How to run astro_traj
---------------------

The main product of this package is the command-line executable `LIGOtraj`

To run an analysis, you need to specify a file that holds the posterior samples from your favorite PE run, an effective radius of your galaxy (in kpc), and an offset for the event (in kpc):

.. code-block:: bash

   $ LIGOTraj --samples 'path_to_samples' --effective-radius 'float' --offset 'float'

You can specifty the number of trials and the name of your output file using flags. You can also alter the galaxy and telescope properties or add a new galaxy/telescope using dictionaries in the constr_dict.py file. 

.. code-block:: bash

   $ LIGOTraj --samples 'path_to_samples' --effective-radius 'float' --offset 'float' --galaxy 'NGC' --telescope 'ESO' --trails 100 --outfile 'path_to_output'

For a full list of command-line argument and options, run

.. code-block:: bash

   $ LIGOTraj --help 

For more details see :ref:`command-line`.

Package documentation
---------------------

Please consult these pages for more details on using astro_traj:

.. toctree::
   :maxdepth: 1

   command-line/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
