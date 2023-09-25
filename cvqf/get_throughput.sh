#!/bin/bash

FILE='/home/ubuntu/filter_data/cvqf/cvqf_10.dat'

rm -rf $FILE

for i in 10
do
	for j in {10..90..10}
	do
		sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
  	sleep 10
		echo "[slot: $i, zipf_const: 0, load_factor: $j]" >> $FILE
		./main $i $j 0 >> $FILE
		echo "" >> $FILE
	done
done
