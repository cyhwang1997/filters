#!/bin/bash

FILE='/home/ubuntu/filter_data/qf_17.dat'

rm -rf $FILE

for i in {0..95..5}
do
  echo "[Load Factor: $i]" >> $FILE
	./test 17 $i >> $FILE
  echo "" >> $FILE
done
