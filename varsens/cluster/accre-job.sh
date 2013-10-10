#!/bin/bash
#PBS -l nodes=1:ppn=1:centos6
#PBS -l walltime=1:00:00
#PBS -l mem=1GB
#PBS -j oe
#PBS -l group=lola

# $PBS_ARRAYID is the number of the block (starts from 1)

cd egfr

python varsens/compute_objective.py varsens/samples/egrf-batch-$PBS_ARRAYID.csv