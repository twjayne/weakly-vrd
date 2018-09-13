#!/bin/bash

# Get width of terminal
COLS=$(( $(tput cols) - 2 ))
# Get nvidia-smi data
SMI=$(nvidia-smi | awk 'x == 1 {print} /PID/ {x = 1}' | tr ' ' $'\t' | tr -s $'\t')
# Get ps data
PS=$(cut -f3 <<< "$SMI" | grep -P '[0-9]+' | xargs ps -o 'pid,user,pcpu,pmem,bsdtime,args' -p)

BAR=$(printf "%${COLS}s" \*)
BAR="${BAR// /*}"
# Print HEADER
echo "$BAR"
printf '%3s %8s ' GPU GMEM
head -n1 <<< "$PS"

# Print BODY ROWS
# Declare IFS to avoid default behaviour of discarding initial whitespace
while IFS=$'\n' read -r row; do
	pid=$(sed 's/^ \+//g' <<< "$row" | tr -s ' ' | cut -d' ' -f1)
	while read -a nvidia; do
		printf '%3s %8s %s\n' "${nvidia[0]}" "${nvidia[1]}" "$row" | cut -c1-$COLS
	done < <(grep $pid <<< "$SMI" | cut -f2,6)
done < <(tail -n+2 <<< "$PS")
echo "$BAR"
