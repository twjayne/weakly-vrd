#!/bin/bash


nvidia-smi | awk 'x == 1 {print $0}; /PID/ {x = 1}' \
| grep -P '\d' \
| tr -s ' ' \
| cut -d' ' -f3 \
| xargs ps -l -p