#!/bin/bash
#PBS -l nodes=1:ppn=1:centos6
#PBS -l walltime=1:00:00
#PBS -l mem=1GB
#PBS -j oe
#PBS -l group_list=lola

# $PBS_ARRAYID is the number of the block (starts from 1)

python run_egfr.py $PBS_ARRAYID