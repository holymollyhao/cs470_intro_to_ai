LOG_PREFIX="submission_source"

#INITIAL parameters for running model
# GPUS=(0 1 2 3 4 5 6 7)
GPUS=(1)
NUM_GPUS=${#GPUS[@]}
SEED="0 1 2"
i=0

sleep 1 # prevent mistake

#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=1
  local num_max_jobs=${1:-$default_num_jobs}
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

train_bert_based(){
  MODEL="bert distilbert"
  DATASET="finefood imdb sst-2"
  epoch="1"
  lr="0.00005"
  method="Src"
  for seed in $SEED; do
    for dataset in $DATASET; do
      for model in $MODEL; do
        python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
                        --dataset $dataset \
                        --method ${method} \
                        --src train \
                        --tgt test \
                        --epoch $epoch \
                        --lr ${lr} \
                        --model $model \
                        --seed $seed \
                        --log_prefix ${LOG_PREFIX}_${seed}_epoch${epoch}_lr${lr}_model${model} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
        wait_n
        i=$((i + 1))
      done
    done
  done
}
train_bart(){
  model="bart"
  DATASET="finefood imdb sst-2"
  lr="0.00001"
  method="Src"
  for dataset in $DATASET; do
    python main.py  --parallel \
                    --dataset $dataset \
                    --method ${method} \
                    --src train \
                    --tgt test \
                    --epoch 1 \
                    --lr ${lr} \
                    --model $model \
                    --seed $SEED \
                    --log_prefix ${LOG_PREFIX}_${SEED}_epoch1_lr${lr}_model${model} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
    wait
  done
}

train_bert_based
#wait
#train_bart