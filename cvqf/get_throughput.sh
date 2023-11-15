#!/bin/bash

FILE="/home/ubuntu/filter_data/data/zipf/cvqf/cvqf26_9.dat"

for i in {1..10..1}
do
	echo "[[$i] zipf_const: 0.9, load_factor: 90]" >> $FILE
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 	sleep 10
	./main 26 90 0.9 >> $FILE
	echo "" >> $FILE
done
