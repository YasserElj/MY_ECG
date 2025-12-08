# Transfer Checkpoints

## From Atlas (local machine) - Pull from HPC

```bash
# Copy single checkpoint
scp yasser.eljarida@<HPC_HOST>:/home/yasser.eljarida/lustre/recsys-y8xnv0gztug/users/yasser.eljarida/code/MY_ECG/LeJEPA-ECG/checkpoints/chkpt_30000.pt ~/hdd1/code/ecg/LeJEPA-ECG/checkpoints/

# Copy all checkpoints
scp yasser.eljarida@<HPC_HOST>:/home/yasser.eljarida/lustre/recsys-y8xnv0gztug/users/yasser.eljarida/code/MY_ECG/LeJEPA-ECG/checkpoints/*.pt ~/hdd1/code/ecg/LeJEPA-ECG/checkpoints/
```

## From HPC - Push to Atlas

```bash
# Copy single checkpoint
scp /home/yasser.eljarida/lustre/recsys-y8xnv0gztug/users/yasser.eljarida/code/MY_ECG/LeJEPA-ECG/checkpoints/chkpt_30000.pt yasser@10.50.28.67:~/hdd1/code/ecg/LeJEPA-ECG/checkpoints/

# Copy all checkpoints
scp /home/yasser.eljarida/lustre/recsys-y8xnv0gztug/users/yasser.eljarida/code/MY_ECG/LeJEPA-ECG/checkpoints/*.pt yasser@10.50.28.67:~/hdd1/code/ecg/LeJEPA-ECG/checkpoints/
```

```bash
python -m finetune \
  --data-dir "../dataset/ptb-xl" \
  --encoder "../LeJEPA-ECG/checkpoints/chkpt_30000.pt" \
  --out "LeJepa_checkpoints_finetune_rhythm" \
  --config "linear" \
  --task "rhythm"

```

Replace `<HPC_HOST>` with your HPC login node hostname.

