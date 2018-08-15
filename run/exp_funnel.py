
import torch
import funnel_runner
from classifier import split_model

import sys
sys.argv.append('--geom')
sys.argv.append('1000 700 ;  ; 1000 700 300 70' )
exp = funnel_runner.Runner()
exp.setup()
exp.train()
