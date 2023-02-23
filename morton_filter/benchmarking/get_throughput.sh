#!/bin/bash

FILE='/home/cyhwang/filter_data/mf_17insert.dat'

rm -rf $FILE

for i in 20 50 90
do
  echo -n "$i " >> $FILE
	./benchmark $i | grep Insertion | awk '{print $3;}' >> $FILE
done
