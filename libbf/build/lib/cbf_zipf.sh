#!/bin/bash

FILE1="/home/ubuntu/filter_data/data/0109_cbf/cbf17_9.dat"
FILE2="/home/ubuntu/filter_data/data/0109_cbf/cbf26_9.dat"

rm -rf $FILE1 $FILE2

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
echo $TIME >> $FILE1

for i in {1..10..1}
do
  echo "[[$i] workload: zipf 0.9, load_factor: 95, size: 17]" >> $FILE1
  sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
  sleep 10
  ./main 131072 14 10 95 0.9 55930880 >> $FILE1 #2^17 slots, 14 bits for counter, 10 hash functions, 95% load factor, zipf 0.9, cvqf range
  echo "" >> $FILE1
done

echo "cbf 17 zipf 0.9 done"

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
echo $TIME >> $FILE2

for i in {1..10..1}
do
  echo "[[$i] workload: zipf 0.9, load_factor: 95, size: 26]" >> $FILE2
  sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
  sleep 10
  ./main 67108864 22 16 95 0.9 28633128960 >> $FILE2 #2^26 slots, 22 bits for counter, 16 hash functions, 95% load factor, zipf 0.9, cvqf range
  echo "" >> $FILE2
done

