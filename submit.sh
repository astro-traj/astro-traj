#!/bin/bash
#MSUB -A b1011
#MSUB -q ligo
#MSUB -l walltime=2:00:00:00
#MSUB -l nodes=1:ppn=1
#MSUB -l partition=quest6
#MSUB -N LIGOTraj
#MOAB -W umask=022
#MSUB -j oe
#MSUB -d /projects/b1011/mzevin/progenitor/

Nsys=1000000

module load python
PATH=/software/Modules/3.2.9/bin:/usr/lib64/qt-3.3/bin:/opt/moab/bin:/opt/moab/sbin:/opt/mam/bin:/opt/mam/sbin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/lpp/mmfs/bin:/opt/ibutils/bin:/opt/torque6/bin:/opt/torque6/sbin:/home/mjz672/bin:/home/mjz672/.local/bin:/home/mjz672/.local/bin

LIGOTraj --grb ${1} --trials ${Nsys} --outfile output/${1}_${MOAB_JOBARRAYINDEX}
