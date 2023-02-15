#!/bin/bash

INPUT=input.csv
PLATFORM=${CL_PLATFORM_ID:-0}
DEVICE=${CL_DEVICE_ID:-0}
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read o1 o2 o3 o4 o5
do
  echo "Running: $o1 (M=$o2, N=$o3, K=$o4, beta=$o5)"
  LD_LIBRARY_PATH=./../build:$LD_LIBRARY_PATH ./smm $o2 $o3 $o4 $o5 $PLATFORM $DEVICE
done < $INPUT

