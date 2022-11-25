LOG_PREFIX="221120_source" ## after normalization

#INITIAL parameters for running model
# GPUS=(0 1 2 3 4 5 6 7)
GPUS=(0)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=1
  local num_max_jobs=1
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

train_bert_based(){
  MODEL="bert distilbert"
  DATASET="finefood imdb sst-2"
  epoch="1"
  lr="0.00002"
  method="Src"
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
                      --seed $SEED \
                      --log_prefix ${LOG_PREFIX}_${SEED}_epoch${epoch}_lr${lr}_model${model} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
      wait_n
      i=$((i + 1))
    done
  done

}

train_bert_based
