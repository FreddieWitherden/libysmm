#!/bin/bash

rm output.csv
it=0
val=3
echo "M","K","N","diff","GLOP/s","GiB/s" >> output.csv
bash run.sh |
  while IFS= read -r line
  do 
	it=$[$it + 1]	
	#echo $line | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' >> output.csv
	read -r line1
	read -r line2
	read -r line3
	read -r line4
	read -r line5
	echo $line,$line1,$line2,$line3,$line4,$line5, >> output.csv
	
	if [[ $it -eq $val ]]; then
		it=0
	fi
  done
