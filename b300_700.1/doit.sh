#!/bin/bash

j=0
while [ $j -lt 2000 ]; do
    tspstart=`date +%d.%m.%Y-%H:%M:%S`
    tsp -i b300_700.1_${j}.json heuristic greedy | \
    tsp improve 2opt | \
    tsp -o b300_700.1_${j}_imp.json improve locenum -w 200 -s 100
    tspimprove=`date +%d.%m.%Y-%H:%M:%S`
    tsp -i b300_700.1_${j}_imp.json -o b300_700.1_${j}_sol.json solve bac
    tspend=`date +%d.%m.%Y-%H:%M:%S`
    echo $j $tspstart $tspimprove $tspend >> runtimes
    j=$[j+1]
done
