#!/bin/bash

dirs=$(ls | grep "byXi")
ext=".byXi"
echo -e "L\tGrid\tXi\tEnergy\tForce"
for dir in $dirs
do
    files=$dir/*$ext
    for file in $files
    do
	if [ -f $file ] 
	then
	    L=$(echo $file | cut -d 'L' -f 2 | cut -d '-' -f 2 | cut -d '_' -f 1)
	    grid=$(echo $file | cut -d 'd' -f 3 | cut -d '-' -f 2 | cut -d 'b' -f 1 | cut -d '.' -f 1,2)
	    Xi=$(echo $file | cut -d 'i' -f 3 | cut -d '-' -f 2 | cut -d '_' -f 1)
	    line=$(cat $file | grep -v "#" | awk '{print $3"\t"$4}')
	    if [ -z "$line" ]
	    then
		echo -e $L$'\t'$grid$'\t'$Xi'\t0\t0'
	    else
		echo -e $L$'\t'$grid$'\t'$Xi'\t'$line
	    fi
	fi
    done
done
