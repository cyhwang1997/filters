#!/bin/bash

sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
sleep 10
./main 17 90 0
