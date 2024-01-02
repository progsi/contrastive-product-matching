#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
LR=$2
TEMP=$3
AUG=$4
TRAINFILE="shs100k_svL-train"
TESTFILE="${TRAINFILE//train/gs}"
PREAUG=$6
PROCESSED="shs100k_1000-train"


python run_finetune_siamese.py \
	--model_pretrained_checkpoint /home/repos/contrastive-product-matching/reports/contrastive/shs100k-$TRAINFILE-clean-$PREAUG$BATCH-$LR-$TEMP-roberta-base/pytorch_model.bin \
    --do_train \
	--dataset_name=shs100k_1000 \
    --train_file /home/repos/contrastive-product-matching/data/interim/shs100k_1000/$TRAINFILE.json.gz \
	--validation_file /home/repos/contrastive-product-matching/data/interim/shs100k_1000/$TRAINFILE.json.gz \
	--test_file /home/repos/contrastive-product-matching/data/interim/shs100k_1000/$TESTFILE.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir /home/repos/contrastive-product-matching/reports/contrastive-ft-siamese/shs100k-$TRAINFILE-$AUG$BATCH-$PREAUG$LR-$TEMP-frozen-roberta-base/ \
	--per_device_train_batch_size=64 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
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