#!/bin/bash
# Usage: ./metric_all.sh <model_path> <mask_path> <iteration>
echo "Running metrics for model: ${1}, mask: ${2}, iteration: ${3}"
python render.py -m ${1} --skip_train --loaded_pth ${1}/chkpnt${3}.pth
python metrics.py -m ${1}
python metrics_mask.py -m ${1} --mask_path ${2} --gt_path ${1}/test/ours_None/gt
python metric_fps.py -m ${1}
python metric_number.py -m ${1} --skip_train --loaded_pth ${1}/chkpnt${3}.pth
python metric_storage.py ${1} ${3}
echo "Running summary for model: ${1}, mask: ${2}, iteration: ${3}" >> summary.txt
python summary.py ${1} ${3} >> summary.txt