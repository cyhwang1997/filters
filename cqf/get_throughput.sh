#!/bin/bash

FILE='/home/ubuntu/filter_data/data/1219/cqf17_kosarak.dat'
FILE2='/home/ubuntu/filter_data/data/1219/cqf26_kosarak.dat'

rm -rf $FILE $FILE2

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
echo $TIME > $FILE

for i in {1..10..1}
do
	echo "[[$i]workload: kosarak, load factor: 95, size: 17]" >> $FILE
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
  sleep 10
	/home/ubuntu/cqf/test 17 95 -1 >> $FILE
  echo "" >> $FILE
done

echo "17 kosarak done"

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
echo $TIME >> $FILE2

for i in {1..10..1}
do
	echo "[[$i]workload: kosarak, load factor: 95, size: 26]" >> $FILE2
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
  sleep 10
	/home/ubuntu/cqf/test 26 95 -1 >> $FILE2
  echo "" >> $FILE2
done

echo "26 kosarak done"

#TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
#echo $TIME >> $FILE3

#for i in {1..10..1}
#do
#	echo "[[$i]workload: caida, load factor: 95, size: 17]" >> $FILE3
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#  sleep 10
#	/home/ubuntu/cqf/test 17 95 -1 >> $FILE3
#  echo "" >> $FILE3
#done

#TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
#echo $TIME >> $FILE4

#for i in {1..10..1}
#do
#	echo "[[$i]workload: zipf 0.9, load factor: 95, size: 17]" >> $FILE4
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#  sleep 10
#	/home/ubuntu/cqf/test 17 95 0.9 >> $FILE4
#  echo "" >> $FILE4
#done
