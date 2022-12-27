#!/bin/bash

INPUT=input.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
echo $o1,$o2,$o3
while read o1 o2 o3
do
  LD_LIBRARY_PATH=./../build:$LD_LIBRARY_PATH ./smm $o1 $o2 $o3
done < $INPUT

