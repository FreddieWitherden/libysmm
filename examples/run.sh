#!/bin/bash

INPUT=input.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read o1 o2 o3
do
                        ./a.out $o1 $o2 $o3
done < $INPUT



