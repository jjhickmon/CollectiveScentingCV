#!/bin/bash
FILE_NUM=7

# python translate_track_to_datalog.py <<EOF
# $FILE_NUM
# EOF
python step_2__run_scenting_classification.py <<EOF
$FILE_NUM
EOF
python step_3__run_orientation_estimator.py <<EOF
$FILE_NUM
EOF
python step_4__visualize.py <<EOF
$FILE_NUM
EOF