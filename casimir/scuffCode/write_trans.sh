#!/bin/bash

#translation file name
file="Bead.trans"

#translation range/step in microns
minimum=1.0
maximum=1.0
step=1.0

if [ -f $file ]
then
    printf "Removing Old Trans File: "
    rm -rv $file
fi

for i in $(seq $minimum $step $maximum) 
do
    echo "TRANS" $i OBJECT Sphere DISP 0.0 0.0 $i >> $file
done
