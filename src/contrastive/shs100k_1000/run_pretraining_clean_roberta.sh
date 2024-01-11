#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
TASK=$1
BATCH=$2
EPOCHS=$3
LR=5e-05
TEMP=0.07
AUG=all-
TRAINFILE=shs100k_$TASK-train
PROCESSED="shs100k_1000-train.pkl.gz"

python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=shs100k_1000 \
	--clean=True \
    --train_file /data/repos/contrastive-product-matching/data/processed/shs100k_1000/contrastive/$PROCESSED \
	--id_deduction_set /data/repos/contrastive-product-matching/data/interim/shs100k_1000/$TRAINFILE.json.gz \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /data/repos/contrastive-product-matching/reports/contrastive/shs100k-$TRAINFILE-clean-$AUG$BATCH-$LR-$TEMP-roberta-base/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG