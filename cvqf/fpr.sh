#!/bin/bash

FILE="/home/ubuntu/filter_data/data/fpr.dat"

TIME=$(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')
echo $TIME >> $FILE

#uniform
echo "[[$i] workload: uniform, load_factor: 95, size: 17]" >> $FILE
sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#sleep 10
/home/ubuntu/filters/cvqf/main_fpr 17 95 0 >> $FILE
echo "" >> $FILE

#zipf 0.9
echo "[[$i] workload: zipf 0.9, load_factor: 95, size: 17]" >> $FILE
sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#sleep 10
/home/ubuntu/filters/cvqf/main_fpr 17 95 0.9 >> $FILE
echo "" >> $FILE

#zipf 0.99
echo "[[$i] workload: zipf 0.99, load_factor: 95, size: 17]" >> $FILE
sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
#sleep 10
/home/ubuntu/filters/cvqf/main_fpr 17 95 0.99 >> $FILE
echo "" >> $FILE

