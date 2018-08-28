#!/bin/bash

# Get width of terminal
COLS=$(( $(tput cols) - 2 ))
# Get nvidia-smi data
SMI=$(nvidia-smi | awk 'x == 1 {print} /PID/ {x = 1}' | tr ' ' $'\t' | tr -s $'\t')
# Get ps data
PS=$(cut -f3 <<< "$SMI" | grep -P '[0-9]+' | xargs ps -o 'pid,user,pcpu,pmem,bsdtime,args' -p)

# Print HEADER
printf '%3s %7s ' GPU GMEM
head -n1 <<< "$PS"

# Print BODY ROWS
while read -r row; do
	pid=$(tr -s ' ' <<< "$row" | cut -d' ' -f1)
	while read -a nvidia; do
		printf '%3s %7s %s\n' "${nvidia[0]}" "${nvidia[1]}" "$row" | cut -c1-$COLS
	done < <(grep $pid <<< "$SMI" | cut -f2,6)
done < <(tail -n+2 <<< "$PS")
