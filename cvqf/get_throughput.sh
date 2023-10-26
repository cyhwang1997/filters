#!/bin/bash

FILE='/home/ubuntu/filter_data/max_allowed_keys/cvqf17.dat'

rm -rf $FILE

for i in 0 0.9 0.99
do
  echo $i
	for j in {10..90..10}
	do
		sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 		sleep 10
		echo "[zipf_const: $i, load_factor: $j]" >> $FILE
		./main 17 $j $i >> $FILE
		echo "" >> $FILE
	done
done
