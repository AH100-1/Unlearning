#!/bin/bash
set -e

log_path='/data/khw/unlearn_kd_class/ckpt/retain_models'
mkdir -p ${log_path}

cd /data/khw  # 루트로 이동

python -m unlearn_kd_class.scripts.make_control --seed 42 --teacher_model r34 --student_model r18 --unlearn_class 0 > ${log_path}/c1control_r34_r18.log 2>&1 &
python -m unlearn_kd_class.scripts.make_control --seed 42 --teacher_model r50 --student_model r18 --unlearn_class 0 > ${log_path}/c1control_r50_r18.log 2>&1 &
python -m unlearn_kd_class.scripts.make_control --seed 42 --teacher_model r50 --student_model r34 --unlearn_class 0 > ${log_path}/c1control_r50_r34.log 2>&1 &
