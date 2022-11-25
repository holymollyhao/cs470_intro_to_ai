LOG_PREFIX="221119_source_backbone_true" ## after normalization


#INITIAL parameters for running model
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0
DATASETS="sst-2 imdb finefood" #this is fixed for source training
METHODS="Src"

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=6 #12
  local num_max_jobs=6
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}
for EPOCH in $EPOCHS; do
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      if [ "${METHOD}" = "Src" ]; then
        MODEL="bert"
        TGT="test"
        python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
                        --dataset $DATASET \
                        --lr 0.00005 \
                        --method ${METHOD} \
                        --tgt $TGT \
                        --model $MODEL \
                        --epoch 1 \
                        --seed $SEED \
                        --log_prefix ${LOG_PREFIX}_${SEED}_epoch${EPOCH} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}_epoch${EPOCH}.txt &
      fi
      wait_n
      i=$((i + 1))
    done
  done
done
