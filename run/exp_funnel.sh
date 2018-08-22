#!/bin/bash

. 'source' "$@"

COUNT=0

function experiment ()
{
	read -r -a A <<< "$1"
	read -r -a B <<< "$2"
	read -r -a C <<< "$3"
	while read -r scenic; do

		# Build 'geometry' string
		read -r firstfile < <(ls -1 "$scenic/train" | head -n1)
		read -r width < <(python ../util/matlab.py "$scenic/train/$firstfile")
		if ((${#B[@]})); then
			geom="${A[@]} ; $width ${B[@]} ; $(( ${B[-1]} + ${A[-1]} )) ${C[@]}"
		elif ((${#A[@]})); then # if B is empty but not A
			geom="${A[@]} ;                ; $(( $width  + ${A[-1]} )) ${C[@]}"
		else # if A and B are empty
			geom="        ;                ; $(( $width  + ${C[0]}  )) ${C[@]:1}"
		fi

		# Other options
		gpu=$(( $COUNT % 3 ))
		export CUDA_VISIBLE_DEVICES=$gpu 
		outdir="$HOME/data/weakly-vrd/out/noval scenic-from-matlab w-variable-at-concat/geom $geom"

		# Run
		CMD=(\
		python funnel_runner.py \
			"$scenic/train" \
			"$TRAINFNAMES" \
			"$scenic/test" \
			"$TESTFNAMES" \
			--outdir "$outdir" \
			--log "$outdir/out.log" \
			--geom "$geom" \
			--ep 30 \
			--save-best \
			--test_every 100 \
		)
		if (($DRY_RUN)); then
			printf "${CMD[0]}"
			for item in "${CMD[@]:1}"; do
				printf " \"${item}\""
			done
			echo
		else
			CMD=(nohup "${CMD[@]}")
			if (($SILENT)); then
				OUTFILE=/dev/null
			else
				OUTFILE="${OUTFILE:-nohup.out}"
			fi
			if (($OUTFILE)); then # no nohup.out file
				CMD=(nohup "${CMD[@]}")
			fi
			if (($FOREGROUND)); then
				"${CMD[@]}" >>"$OUTFILE"
			else
				"${CMD[@]}" >>"$OUTFILE" &
			fi
		fi
			# >/dev/null &
		COUNT=$(( $COUNT + 1 ))
	done < <(ls -1 -d $HOME/data/sg_dataset/scenic/pca*)
}

experiment '1000 800' '800' '700 70'
experiment '' '' '1000 500 250 70'
experiment '1000 800' '' '700 70'
