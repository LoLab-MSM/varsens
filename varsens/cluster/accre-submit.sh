#!/bin/bash
# Submits a PySB job

usage()
{
cat << EOF
usage: $0 script blocks

EOF
}

if [[ $# -lt 2 ]]; then
	usage
	exit 1
fi
if [[ $# -gt 2 ]]; then
	usage
	exit 1
fi

BLOCKS=$1
HERE=`pwd`

mkdir -p $HERE/results

# Submit jobs as a multiple of blocks
qsub -o $HERE/results -N PySB -t 1-$BLOCKS accre-job.sh

