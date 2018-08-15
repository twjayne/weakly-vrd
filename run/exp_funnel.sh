#!/bin/bash

TRAINFNAMES="$HOME/data/unrel/data/vrd-dataset/image_filenames_train.mat"
TESTFNAMES="$HOME/data/unrel/data/vrd-dataset/image_filenames_test.mat"

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
		gpu=$(( ($COUNT + 1) % 3 ))
		export CUDA_VISIBLE_DEVICES=$gpu 
		outdir="$HOME/data/weakly-vrd/out/geom $geom"

		# Run
		nohup python funnel_runner.py \
			"$scenic/train" \
			"$TRAINFNAMES" \
			"$scenic/test" \
			"$TESTFNAMES" \
			--outdir "$outdir" \
			--log "$outdir/out.log" \
			--geom "$geom" \
			--ep 50 \
			--test_every 100 \
			>/dev/null &

		COUNT=$(( $COUNT + 1 ))
	done < <(ls -1 -d $HOME/data/sg_dataset/scenic/pca*)
}

# experiment '1000 800' '800' '700 70'
experiment '' '' '1000 500 250 70'
experiment '1000 800' '' '700 70'
