#!/bin/bash

DIR="/home/ubuntu/filter_data/data/0109_cache"

echo "l1 cache"
for i in {10..90..10}
do
	FILE="$DIR/cvqfl1_$i.dat"
	echo $i
	for j in {1..10..1}
	do
		echo "[[$j] zipf_const: 0.9, load_factor: $i]" >> $FILE
		sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
	 	sleep 10
		/home/ubuntu/filters/cvqf/main 14 $i 0.9 >> $FILE
		echo "" >> $FILE
	done
done

echo "l2 cache"
for i in {10..90..10}
do
	FILE="$DIR/cvqfl2_$i.dat"
	echo $i
	for j in {1..10..1}
	do
		echo "[[$j] zipf_const: 0.9, load_factor: $i]" >> $FILE
		sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
	 	sleep 10
		/home/ubuntu/filters/cvqf/main 17 $i 0.9 >> $FILE
		echo "" >> $FILE
	done
done

echo "l3 cache"
for i in {10..90..10}
do
	FILE="$DIR/cvqfl3_$i.dat"
	echo $i
	for j in {1..10..1}
	do
		echo "[[1] zipf_const: 0.9, load_factor: $i]" >> $FILE
		sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
	 	sleep 10
		/home/ubuntu/filters/cvqf/main 24 $i 0.9 >> $FILE
		echo "" >> $FILE
	done
done

