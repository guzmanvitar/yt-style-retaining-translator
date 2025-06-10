#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wav2lip-legacy
python "$@"
