#!/bin/bash

DVAL=${1}
LAM=${2}

ulimit -c ${CORE_LIMIT}
set -e
finalRC=1

MYDIR=/afs/slac.stanford.edu/u/xo/dcmoore
OUTDIR=/nfs/slac/g/exo/dcmoore/analysis/force_calc/data

export SCRATCHDIR=/scratch/dcmoore/pull_chi2_v5_$LAM
mkdir -p ${SCRATCHDIR}
cd ${SCRATCHDIR}

gotEXIT()
{
 rm -rf ${SCRATCHDIR}
 exit $finalRC
}
trap gotEXIT EXIT

source ${MYDIR}/setup_trunk.sh

python /nfs/slac/g/exo/dcmoore/analysis/force_calc/force_calc_v2_cyl.py ${DVAL} ${LAM}

scp *.npy ${OUTDIR}

rm -rf ${SCRATCHDIR}

finalRC=0

