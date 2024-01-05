#!/bin/sh
export LD_LIBRARY_PATH=/home/ubuntu/filters/libbf/build/lib:$LD_LIBRARY_PATH
g++ -o main main.cc -I/home/ubuntu/filters/libbf -L./ -lbf -lssl -lcrypto

#./main 131088 12 10 90 0.9 55930880

#./main 131072 13 10 95 0.9 55930880 #2^17 slots, 13 bits for counter, 95% load factor, 0.9 zipf const

#./main 
