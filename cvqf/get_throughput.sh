#!/bin/bash

DIR="/home/ubuntu/filter_data/data/cvqf_overhead/1101_2/cvqf17_"

for i in {1..10..1}
do
	for j in {10..90..10}
	do
		FILE="$DIR$i.dat"
		echo "[zipf_const: 0, load_factor: $j]" >> $FILE
		sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 		sleep 10
		./main 17 $j 0 >> $FILE
		echo "" >> $FILE
	done
done
