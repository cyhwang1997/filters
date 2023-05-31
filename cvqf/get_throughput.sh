#!/bin/bash

FILE='/home/ubuntu/filter_data/cvqf_21.dat'

rm -rf $FILE

for i in {0..95..5}
do
  echo "[Load Factor: $i]" >> $FILE
	./main 21 $i 0 >> $FILE
  echo "" >> $FILE
done
