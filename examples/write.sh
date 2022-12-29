#!/bin/bash

rm output.csv
it=0
val=3
echo "diff","GLOP/s","GiB/s" >> output.csv
bash run.sh |
  while IFS= read -r line
  do 
	it=$[$it + 1]	
	#echo $line | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' >> output.csv
	read -r line2
	read -r line3 
	echo $line,$line2,$line3 >> output.csv
	
	if [[ $it -eq $val ]]; then
		it=0
	fi
  done
