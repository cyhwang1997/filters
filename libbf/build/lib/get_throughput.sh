#!/bin/bash

#FILE1="/home/ubuntu/filter_data/data/1219/cbf17_webdocs.dat"
FILE2="/home/ubuntu/filter_data/data/1219/cbf26_webdocs.dat"
#FILE3="/home/ubuntu/filter_data/data/counting_fpr/cbf17_99.dat"
#FILE4="/home/ubuntu/filter_data/data/counting_fpr/cbf26_99.dat"

#rm -rf $FILE1 $FILE2

#TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
#echo $TIME >> $FILE1

#for i in {1..10..1}
#do
#	echo "[[$i] workload: webdocs, load_factor: 95, size: 17]" >> $FILE1
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
# 	sleep 10
#	./main 131072 14 10 95 -1 55930880 >> $FILE1 #2^14 slots, 14 bits for counter, 95% load factor, 10 hash functions, uniform, cvqf range
#	echo "" >> $FILE1
#done

#echo "cbf 17 webdocs done"

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
echo $TIME >> $FILE2

for i in {7..10..1}
do
	echo $i
	echo "[[$i] workload: webdocs, load_factor: 95, size: 26]" >> $FILE2
	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
 	sleep 10
	./main 67108864 22 16 95 -1 28633128960 >> $FILE2 #2^26 slots, 22 bits for counter, 16 hash functions, 95% load factor, 0.99 zipf const
	echo "" >> $FILE2
done

#echo "cbf 17 0.9 done"

#TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
#echo $TIME >> $FILE3

#for i in {1..10..1}
#do
#	echo $i
#	echo "[[$i] zipf_const: 0.9, load_factor: 95]" >> $FILE3
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
# 	sleep 10
#	./main 131072 14 10 95 0.99 55930880 >> $FILE3 #2^17(131072) slots, 14 bits for counter, 10 hash functions, 95% load factor, 0.99 zipf const
#	echo "" >> $FILE3
#done

#echo "cbf 17 0.99 done"

#for i in {1..10..1}
#do
#	echo $i
#	echo "[[$i] zipf_const: 0.9, load_factor: 95, size: 26]" >> $FILE4
#	sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#	sleep 10
#	./main 67108864 22 16 95 0.9 28633128960 >> $FILE4 #2^17 slots, 14 bits for counter, 95% load factor, 0.99 zipf const
#	echo "" >> $FILE4
#done

#echo "cbf 26 0.99 done"
