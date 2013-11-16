#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l mem=2000mb
#PBS -j oe
#PBS -m a
#PBS -M shawn.garbett@vanderbilt.edu

# $PBS_ARRAYID is the number of the block (starts from 1)

cd ~/earm

python varsens/compute_objective.py /scratch/garbetsp/varsens/samples/earm-batch-$PBS_ARRAYID.csv
