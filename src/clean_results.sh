#!/bin/bash

for e in ~/ray_results/*; do
	size=$(du -s $e | cut -f1)
	if [[ $size -lt 2000 ]]; then
		echo "Delete $e, size $size"
		rm -r $e
	fi
done
