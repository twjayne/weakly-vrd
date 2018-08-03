#!/bin/bash

# Opts:
# 	--ep $ep
# 	--lr $lr
# 	-N ${N}
# 	--noval

# Usage e.g.
# 	LOGD=log/overfit/noval ./demo.sh --ep 30 --lr 0.001 -N 0 --noval

[[ -z $LOGD ]] && LOGD=log/overfit/noval
mkdir -p "$LOGD"

. activate cs231n
python demo.py $@ | tee "$LOGD/N-${N}-ep-${ep}-lr-${lr}.log"
