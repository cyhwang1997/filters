#!/bin/bash

FILE="/home/ubuntu/filter_data/data/real_datasets/caida/cvqf26_caida.dat"

for i in {1..10..1}
do
	echo "[[$i] workload: caida, load_factor: 95]" >> $FILE
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 	sleep 10
	./main 26 95 -1 >> $FILE
	echo "" >> $FILE
done
