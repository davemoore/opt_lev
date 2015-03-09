#!/bin/bash

filedir="/u/ki/kurinsky/GrattaGroup/opt_lev/casimir/scuffCode/CubeSphereSG/"
geofile="BeadSG.scuffgeo"
meshfiles="SphereSG.geo CubeSG.geo"

filestr="BeadCubeSG"

if [ $# -gt 1 ]
then
    L=$1
    gridding=$2
    #translation file name
    filebase=$filestr"_L-"$L"_grid-"$gridding
    
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

    echo "scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --energy --zforce"
    scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --energy --zforce

    rm -v *.msh
    rm -v *.geo

    tail -n 1 *.out >> ../results.txt

else
    echo "Not enough arguments"
    echo "Calling Sequence: "$0" xi distance gridding"
fi