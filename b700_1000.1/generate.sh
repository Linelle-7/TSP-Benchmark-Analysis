#!/bin/bash

j=0
while [ $j -lt 2000 ]; do
    n=`echo $((RANDOM%300+700))`
    tsp -o b700_1000.1_${j}.json generate -n $n
    j=$[j+1]
done
