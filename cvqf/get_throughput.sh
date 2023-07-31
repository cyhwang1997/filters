#!/bin/bash

FILE='/home/ubuntu/filter_data/cvqf_26_skew.dat'

rm -rf $FILE

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
do
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
  sleep 10
	echo "[zipf_const: $i]" >> $FILE
	./main 26 90 $i >> $FILE
	echo "" >> $FILE
done
