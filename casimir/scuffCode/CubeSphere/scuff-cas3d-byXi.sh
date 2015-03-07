#!/bin/bash

filedir="../../"
geofile="Bead_DL.scuffgeo"
meshfiles="Sphere SurfaceDistanceMeshed"

filestr="Bead_Xi-"

if [ $# -gt 2 ]
then
    Xi=$1
    L=$2
    gridding=$3
    #translation file name
    filebase=$filestr$Xi"_L-"$L"_grid-"$gridding
    
    if [ -d $filebase ]
    then
	rm -rvf $filebase
    fi
    mkdir $filebase
    cd $filebase
    
    file=$filebase".trans"
    if [ -f $file ]
    then
        printf "Removing Old Trans File: "
        rm -rv $file
    fi

    echo "TRANS" $1 OBJECT Sphere DISP 0.0 0.0 $L >> $file
    cp $filedir/$geofile ./
    for mshfile in $meshfiles
    do
	cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' > ./$mshfile
	gmsh -2 $mshfile
    done

    echo "scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --energy --zforce --Xi $Xi"
    scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --energy --zforce --Xi $Xi

else
    echo "Not enough arguments"
    echo "Calling Sequence: "$0" xi distance gridding"
fi
