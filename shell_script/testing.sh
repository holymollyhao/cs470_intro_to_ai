LOG_PREFIX="221117_ablation_tests" ## after normalization
METHODS="dattaprompttune"

#INITIAL parameters for running model
GPUS=(2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0

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

DATASETS="finefood" #this is fixed for source training
#ADAPT_TYPE="all bn ln bnln emb"
LRs="0.3 0.1 0.001 0.00001"
for DATASET in $DATASETS; do
  for METHOD in $METHODS; do
    for lr in $LRs; do
      if [ "${METHOD}" = "ttaprompttune" ]; then
          ADAPT_TYPE="all"
      elif [ "${METHOD}" = "ln_tent" ]; then
          ADAPT_TYPE="ln"
      elif [ "${METHOD}" = "dattaprompttune" ]; then
          ADAPT_TYPE="all"
      fi
      for adapt_type in $ADAPT_TYPE; do
        MODEL="bert"
        SRC="train"
        TGT="test"
        uex=16
        memsize=16
        python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
                        --dataset $DATASET \
                        --method ${METHOD} \
                        --src $SRC \
                        --tgt $TGT \
                        --online \
                        --epoch 1 \
                        --lr ${lr} \
                        --update_every_x ${uex} \
                        --memory_size ${memsize} \
                        --load_checkpoint_path /home/twkim/git/tetra/log/sst-2/Src/tgt_test/221114_initial_gen_0/cp/cp_last.pth.tar \
                        --model $MODEL \
                        --seed $SEED \
                        --adapt_type $adapt_type \
                        --log_prefix ${LOG_PREFIX}_${SEED}_lr${lr}_memsize${memsize}_uex${uex}_type${adapt_type} \
                      2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
        i=$((i + 1))
      done
      wait_n
    done
  done
done


#main.py
#--gpu_idx 1
#--lr 0.2
#--dataset finefood
#--method ttaprompttune
#--src test --tgt train
#--online
#--epoch 1
#--model bert
#--seed 0
#--update_every_x 16
#--memory_size 16
#--load_checkpoint_path /home/twkim/git/tetra/log/sst-2/Src/tgt_test/221114_initial_gen_0/cp/cp_last.pth.tar
#--log_prefix debug
#--adapt_type all