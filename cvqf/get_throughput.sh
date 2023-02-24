#!/bin/bash

FILE='/home/cyhwang/filter_data/cvqf_17insert.dat'

rm -rf $FILE

for i in 20 50 90
do
  echo "[Load Factor: $i]" >> $FILE
	./main 17 $i 0 >> $FILE
  echo "" >> $FILE
done
