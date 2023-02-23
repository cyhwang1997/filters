#!/bin/bash

FILE='/home/cyhwang/filter_data/cf_17insert.dat'

rm -rf $FILE

for i in 20 50 90
do
  echo -n "$i " >> $FILE
	./test 17 $i | grep Insertion | awk '{print $3;}' >> $FILE
done
