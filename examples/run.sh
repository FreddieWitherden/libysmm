#!/bin/bash

INPUT=input.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read o1 o2 o3
do
  echo "Running: M="$o1", N="$o2", K="$o3
  LD_LIBRARY_PATH=./../build:$LD_LIBRARY_PATH ./smm $o1 $o2 $o3
done < $INPUT

