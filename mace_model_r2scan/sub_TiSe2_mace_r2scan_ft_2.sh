#!/bin/bash
if [ -z "${1}" ]; then
echo "Must set number of CUDA device. Use nvidia-smi to get numbers of available devices!"
exit -1
fi
export CUDA_VISIBLE_DEVICES="$1"
echo $CUDA_VISIBLE_DEVICES
num_threads=1
export OMP_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads
PYPATH=/home/lukab/miniconda3/envs/mace_cueq/bin/

# =============================================================================
# Multihead-replay finetuning: PBE TiSe2 model (TiSe2_model_6) -> r2SCAN
#
# Two heads get trained together:
#   pt_head  - replay of the ORIGINAL PBE data (--pt_train_file). Keeps the
#                model from forgetting PBE while it learns the new level.
#   default  - the new r2SCAN data (--train_file). This is the head you will
#                actually use afterwards.
#
# EDIT BEFORE RUNNING:
#   FOUNDATION_MODEL below must point at the .model file your PBE run
#   produced (ls the old run's directory - normally just TiSe2_model_6.model;
#   do not point this at a _compiled file, those cannot be finetuned further).
# =============================================================================
FOUNDATION_MODEL="./TiSe2_model_6.model"   # <-- EDIT THIS

$PYPATH/mace_run_train \
    --name="TiSe2_r2scan_mh_ft_2" \
    --foundation_model="${FOUNDATION_MODEL}" \
    --multiheads_finetuning=True \
    --pt_train_file="./TiSe2_frames_train_MACE.xyz" \
    --pt_valid_file="./TiSe2_frames_test_MACE.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --train_file="./TiSe2_r2scan_train.xyz" \
    --valid_fraction=0.05 \
    --test_file="./TiSe2_r2scan_test.xyz" \
    --E0s="{22: -9.043763, 34: -18.087526}" \
    --config_type_weights='{"Default":1.0}' \
    --default_dtype="float64" \
    --batch_size=5 \
    --max_num_epochs=500 \
    --swa \
    --start_swa=400 \
    --ema \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --save_cpu
echo "Done."
