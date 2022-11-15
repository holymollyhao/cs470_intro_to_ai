LOAD_PREFIX="221114_initial_gen"
LOG_BASE_PATH="/home/twkim/git/tetra/log"
METHODS="Src prompttune ttaprompttune"


#INITIAL parameters for running model
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0
DATASETS="tomatoes finefood" #this is fixed for source training


sleep 1 # prevent mistake

#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=1 #12
  local num_max_jobs=7
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}
### Test without any adaptation, for checking generalizablity
no_adapt_naive_src() {
  LOG_PREFIX="221115_no_adapt"
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      if [ "${METHOD}" = "Src" ]; then
        EPOCH=0
        MODEL="bert"
        TGT="test"
      elif [ "${METHOD}" = "finetune" ]; then
        continue
      elif [ "${METHOD}" = "prompttune" ]; then
        EPOCH=0
        MODEL="bert"
        TGT="test"
      elif [ "${METHOD}" = "ttaprompttune" ]; then
        EPOCH=0
        MODEL="bert"
        TGT="test"
      fi

      # load path : /home/twkim/git/tetra/log/sst-2/prompttune/tgt_test/221114_initial_gen_0/cp
      python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
                      --method ${METHOD} \
                      --tgt $TGT \
                      --epoch $EPOCH \
                      --model $MODEL \
                      --seed $SEED \
                      --load_checkpoint_path /${LOG_BASE_PATH}/sst-2/${METHOD}/tgt_test/${LOAD_PREFIX}_${SEED}/cp/cp_last.pth.tar \
                      --log_prefix ${LOG_PREFIX}_from${METHOD}_${SEED}_epoch${EPOCH} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
      i=$((i + 1))
      wait_n
    done
  done
}

#### Finetune + Naive prompt tuning with frozen src model
adapt_with_naive_src() {
  LOG_PREFIX="221115_naive_src_adapt"
  EPOCHS="1 3 5"
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      for EPOCHS in $EPOCH; do
        if [ "${METHOD}" = "Src" ]; then
          continue
        elif [ "${METHOD}" = "finetune" ]; then
          MODEL="bert"
          TGT="test"
        elif [ "${METHOD}" = "prompttune" ]; then
          MODEL="bert"
          TGT="test"
        elif [ "${METHOD}" = "ttaprompttune" ]; then
          MODEL="bert"
          TGT="test"
        fi

        # load path : /home/twkim/git/tetra/log/sst-2/prompttune/tgt_test/221114_initial_gen_0/cp
        python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
                        --method ${METHOD} \
                        --tgt $TGT \
                        --epoch $EPOCH \
                        --model $MODEL \
                        --seed $SEED \
                        --load_checkpoint_path /${LOG_BASE_PATH}/sst-2/Src/tgt_test/${LOAD_PREFIX}_${SEED}/cp/cp_last.pth.tar \
                        --log_prefix ${LOG_PREFIX}_fromSrc_${SEED}_epoch${EPOCH} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
        i=$((i + 1))
        wait_n
      done
    done
  done
}

### Finetune + Adaptation from pretrained method
adapt_with_pretrained_src() {
  LOG_PREFIX="221115_pretrained_src_adapt"
  EPOCHS="1 3 5"
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      for EPOCH in $EPOCHS; do
        if [ "${METHOD}" = "Src" ]; then
          continue
        elif [ "${METHOD}" = "finetune" ]; then
          continue
        elif [ "${METHOD}" = "prompttune" ]; then
          MODEL="bert"
          TGT="test"
        elif [ "${METHOD}" = "ttaprompttune" ]; then
          MODEL="bert"
          TGT="test"
        fi

        # load path : /home/twkim/git/tetra/log/sst-2/prompttune/tgt_test/221114_initial_gen_0/cp
        python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
                        --method ${METHOD} \
                        --tgt $TGT \
                        --epoch $EPOCH \
                        --model $MODEL \
                        --seed $SEED \
                        --load_checkpoint_path /${LOG_BASE_PATH}/sst-2/${METHOD}/tgt_test/${LOAD_PREFIX}_${SEED}/cp/cp_last.pth.tar \
                        --log_prefix ${LOG_PREFIX}_from${METHOD}_${SEED}_epoch${EPOCH} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
        i=$((i + 1))
        wait_n
        done
    done
  done
}

#no_adapt_naive_src
#adapt_with_naive_src
adapt_with_pretrained_src