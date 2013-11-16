#!/bin/bash
# Submits a PySB job

usage()
{
cat << EOF
usage: $0 blocks

EOF
}

if [[ $# -lt 1 ]]; then
	usage
	exit 1
fi
if [[ $# -gt 1 ]]; then
	usage
	exit 1
fi

BLOCKS=$1
HERE=`pwd`

mkdir -p $HERE/results

# Submit jobs as a multiple of blocks
qsub -o $HERE/results -N PySB -t $BLOCKS accre-job.sh

