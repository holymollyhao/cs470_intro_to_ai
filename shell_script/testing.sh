LOG_PREFIX="221119_bart_tests" ## after normalization
METHODS="dattaprompttune ln_tent"

#INITIAL parameters for running model
# GPUS=(2 3 4 5 6)
GPUS=(0)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0
n_tokens=30

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=1 #12
  local num_max_jobs=1
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}
EPOCHS="1"
DATASETS="finefood" #this is fixed for source training
#ADAPT_TYPE="all bn ln bnln emb"
LRs="0.3 0.2 0.1 0.001 0.00001 0.000001"
for EPOCH in $EPOCHS; do
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      for lr in $LRs; do
        if [ "${METHOD}" = "ttaprompttune" ]; then
            ADAPT_TYPE="all all_ln_bn"
        elif [ "${METHOD}" = "ln_tent" ]; then
            ADAPT_TYPE="ln"
        elif [ "${METHOD}" = "dattaprompttune" ]; then
            ADAPT_TYPE="all embed"
        fi
        for adapt_type in $ADAPT_TYPE; do
          MODEL="bart"
          SRC="train"
          TGT="test"
          uex=4
          memsize=4
          python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
                          --dataset $DATASET \
                          --method ${METHOD} \
                          --src $SRC \
                          --tgt $TGT \
                          --online \
                          --epoch $EPOCH \
                          --lr ${lr} \
                          --update_every_x ${uex} \
                          --memory_size ${memsize} \
                          --model $MODEL \
                          --seed $SEED \
                          --adapt_type $adapt_type \
                          --n_tokens $n_tokens \
                          --log_prefix ${LOG_PREFIX}_${SEED}_epoch${EPOCH}_lr${lr}_memsize${memsize}_uex${uex}_type${adapt_type}_tokens${n_tokens} \
                        2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
done

#                          --load_checkpoint_path /home/twkim/git/tetra/log/sst-2/Src/tgt_test/221114_initial_gen_0/cp/cp_last.pth.tar \

