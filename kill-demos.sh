#!/bin/bash

function search_and_kill ()
{
	echo grepping $1
	ps ux | grep "$1" | grep -v grep | grep -v kill-demos | tr -s ' ' | cut -d' ' -f2 | while read pid; do
		ps -p $pid | tail -n1
		[[ -n $pid ]] && kill $pid
	done
}

if (($#)); then
	for arg in "${@:1}"; do
		search_and_kill "$arg"
	done
else
	search_and_kill 'run.sh'
	search_and_kill 'python demo'
fi
