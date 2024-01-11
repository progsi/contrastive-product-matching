#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
TASK=$1
BATCH=$2
LR=5e-05
TEMP=0.07
AUG=all-
PREAUG=all-
TRAINFILE=shs100k_$TASK-train
TESTFILE="${TRAINFILE//train/gs}"
PROCESSED="shs100k_1000-train"


python run_finetune_siamese.py \
	--model_pretrained_checkpoint /data/repos/contrastive-product-matching/reports/contrastive/shs100k-$TRAINFILE-clean-$PREAUG$BATCH-$LR-$TEMP-roberta-base/pytorch_model.bin \
    --do_train \
	--dataset_name=shs100k_1000 \
    --train_file /data/repos/contrastive-product-matching/data/interim/shs100k_1000/$TRAINFILE.json.gz \
	--validation_file /data/repos/contrastive-product-matching/data/interim/shs100k_1000/$TRAINFILE.json.gz \
	--test_file /data/repos/contrastive-product-matching/data/interim/shs100k_1000/$TESTFILE.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir /data/repos/contrastive-product-matching/reports/contrastive-ft-siamese/shs100k-$TRAINFILE-$AUG$BATCH-$PREAUG$LR-$TEMP-frozen-roberta-base/ \
	--per_device_train_batch_size=64 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=30 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \