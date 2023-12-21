#!/bin/bash

FILE="/home/ubuntu/filter_data/data/1219/cvqf17_kosarak.dat"
#sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#sleep 10
#./main 26 95 -1

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S') 
echo $TIME >> $FILE

for i in {1..10..1}
do
	echo "[[$i] workload: kosarak, load_factor: 95, size: 17]" >> $FILE
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 	sleep 10
	./main 17 95 -1 >> $FILE
	echo "" >> $FILE
done

echo "17 kosarak done"

FILE="/home/ubuntu/filter_data/data/1219/cvqf26_kosarak.dat"

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S') 
echo $TIME >> $FILE

for i in {1..10..1}
do
	echo "[[$i] workload: kosarak, load_factor: 95, size: 26]" >> $FILE
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 	sleep 10
	./main 26 95 -1 >> $FILE
	echo "" >> $FILE
done

echo "26 kosarak done"

#FILE="/home/ubuntu/filter_data/data/1212/cvqf/cvqf26_9.dat"

#TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S') 
#echo $TIME >> $FILE

#for i in {1..10..1}
#do
#	echo "[[$i] workload: zipf 0.9, load_factor: 95, size: 26]" >> $FILE
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
# 	sleep 10
#	./main 26 95 0.9 >> $FILE
#	echo "" >> $FILE
#done

#echo "26 zipf 0.9 done"

#FILE="/home/ubuntu/filter_data/data/1212/cvqf/cvqf26_caida.dat"

#TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S') 
#echo $TIME >> $FILE

#for i in {1..10..1}
#do
#	echo "[[$i] workload: caida, load_factor: 95, size: 26]" >> $FILE
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
# 	sleep 10
#	./main 26 95 -1 >> $FILE
#	echo "" >> $FILE
#done

#echo "26 caida done"

