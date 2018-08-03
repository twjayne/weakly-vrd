#!/bin/bash

function search_and_kill ()
{
	echo grepping $1
	ps ux | grep "$1" | grep -v grep | tr -s ' ' | cut -d' ' -f2 | while read pid; do
		ps -p $pid | tail -n1
		[[ -n $pid ]] && kill $pid
	done
}

search_and_kill 'run.sh'
search_and_kill 'python demo'
