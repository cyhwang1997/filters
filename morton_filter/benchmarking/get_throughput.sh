#!/bin/bash

FILE='/home/ubuntu/filter_data/mf_21.dat'

rm -rf $FILE

for i in {0..95..5}
do
  echo "[Load Factor: $i]" >> $FILE
	./benchmark $i >> $FILE
  echo "" >> $FILE
done
