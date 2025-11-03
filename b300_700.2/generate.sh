#!/bin/bash

j=0
while [ $j -lt 2000 ];do
    n=`echo $((RANDOM%400+300))`
    tsp -o b300_700.2_${j}.json generate -n $n
    j=$[j+1]
done
