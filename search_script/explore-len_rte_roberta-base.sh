export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0

bs=32
# lr=7e-3
dropout=0.1
# psl=20 #128
epoch=80 


for lr in 1e-2 
do
  for psl in  10 30 60 100 128 #8 16 32 64 128
  do 
    for seed in 11 22 33 
    do
     python3 run.py \
        --model_name_or_path roberta-base \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --pre_seq_len $psl \
        --output_dir checkpoints/$DATASET_NAME-$epoch-$lr-$psl/ \
        --overwrite_output_dir \
        --hidden_dropout_prob $dropout \
        --seed $seed \
        --save_strategy no \
        --evaluation_strategy epoch \
        --prefix
    done
  done
done

# python3 search.py $DATASET_NAME roberta
