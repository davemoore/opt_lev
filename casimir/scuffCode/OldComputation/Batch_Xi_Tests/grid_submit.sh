#!/bin/bash

minlLog=-1
maxlLog=1
dlLog=0.5

minXlog=-3
maxXlog=2
dxlog=1.0

minGlog=-1
maxGlog=1
dGlog=0.2

commands="./scuff-cas3d-PECbyXi.sh ./scuff-cas3d-byXi.sh"

if [ $# -gt 0 ]
then
    if [ $1 -eq "1" ]
    then
	minlLog=-1
        maxlLog=-1
    fi
fi

for LLog in $(seq $minlLog $dlLog $maxlLog) 
do
    for XLog in $(seq $minXlog $dxlog $maxXlog)
    do
	for GLog in $(seq $minGlog $dGlog $maxGlog)
	do
	    L=$(awk 'BEGIN { print 10.0^'$LLog' }')
	    X=$(awk 'BEGIN { print 10.0^'$XLog' }')
	    G=$(awk 'BEGIN { print 10.0^'$GLog' }')
	    for cmd in $commands
	    do
		echo "Submitting $cmd L="$L" G="$G" X="$X
		bsub -q xxl $cmd $X $L $G
	    done
	done
    done
done
