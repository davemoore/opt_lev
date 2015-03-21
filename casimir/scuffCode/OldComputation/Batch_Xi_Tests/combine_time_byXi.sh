#!/bin/bash

dirs=$(ls | grep "log")
ext=".log"
echo -e "L\tGrid\tXi\tDuration\tStatus"
for dir in $dirs
do
    files=$dir/*$ext
    for file in $files
    do
	if [ -f $file ] 
	then
	    L=$(echo $file | cut -d 'L' -f 2 | cut -d '-' -f 2 | cut -d '_' -f 1)
            grid=$(echo $file | cut -d 'd' -f 3 | cut -d '-' -f 2 | cut -d 'l' -f 1 | cut -d '.' -f 1,2)
            Xi=$(echo $file | cut -d 'i' -f 2 | cut -d '-' -f 2 | cut -d '_' -f 1)
	    
	    ran=$(grep "ZForce" $file -c)
	    
	    start=$(head -1 $file | cut -d ':' -f 3,4,5)
	    end=$(tail -1 $file | cut -d ':' -f 3,4,5)
	    daystart=$(head -1 $file | cut -d '/' -f 2)
	    dayend=$(tail -1 $file | cut -d '/' -f 2)

	    days=$(expr $dayend - $daystart)
	    hours=$(expr $(echo $end | cut -d ':' -f 1) - $(echo $start| cut -d ':' -f1))
	    mins=$(expr $(echo $end | cut -d ':' -f 2) - $(echo $start| cut -d ':' -f2))
	    secs=$(expr $(echo $end | cut -d ':' -f 3) - $(echo $start| cut -d ':' -f3))

	    if [ $hours -lt 0 ]
	    then
		hours=$(expr $hours + 24)
	    fi
	    duration=$(echo "86400*$days+3600*$hours+60*$mins+$secs" | bc)
	    if [ -z "$start" ]
	    then
		echo 0
	    else
		echo -e $L'\t'$grid'\t'$Xi'\t'$duration'\t'$ran
	    fi
	fi
    done
done
