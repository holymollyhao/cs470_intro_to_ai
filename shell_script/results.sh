LOG_PREFIX="221201_results" ## after normalization

#INITIAL parameters for running model
GPUS=(1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=0


wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=7 #12
  local num_max_jobs=7
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

N_TOKENS=30
EPOCH=1
DATASETS="finefood imdb sst-2"
LR="0.01 0.00001"
METHODS="Src"
UEX="16 32 128"
MODELS="bert distilbert"
METHODS="Src ln_tent dattaprompttune"
SEED=0
for model in $MODELS; do
  for dataset1 in $DATASETS; do
    for dataset2 in $DATASETS; do
      for lr in $LR; do
        for method in $METHODS; do
          for uex in $UEX; do
            if [ "${model}" = "distilbert" ]; then
              load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/submission_source_0_epoch1_lr0.00001_modeldistilbert/cp/cp_last.pth.tar
            elif [ "${model}" = "bert" ]; then
              load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/submission_source_0_epoch1_lr0.00001_modelbert/cp/cp_last.pth.tar
            fi
            if [ "${method}" = "Src" ]; then
              python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
                              --dataset "${dataset1}" \
                              --method "${method}" \
                              --src train \
                              --tgt test \
                              --epoch 0 \
                              --update_every_x ${uex} \
                              --memory_size ${uex} \
                              --model ${model} \
                              --seed $SEED \
                              --n_tokens ${N_TOKENS} \
                              --log_prefix ${LOG_PREFIX}_${SEED}_model_${model}_from_${dataset2}_to_${dataset1} \
                              --load_checkpoint_path ${load_checkpoint_path} \
                            2>&1 | tee raw_logs/from_${dataset2}_to_${dataset1}_${LOG_PREFIX}_${SEED}_job${i}.txt &
              wait_n
              i=$((i + 1))
            else
              if [ "${method}" = "dattaprompttune" ]; then
                  ADAPT_TYPE="all"
              elif [ "${method}" = "ttaprompttune" ]; then
                  ADAPT_TYPE="all"
              elif [ "${method}" = "ln_tent" ]; then
                  ADAPT_TYPE="ln"
              fi
              for adapt_type in $ADAPT_TYPE; do
                python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
                                --dataset "${dataset1}" \
                                --method "${method}" \
                                --src train \
                                --tgt test \
                                --epoch 1 \
                                --update_every_x ${uex} \
                                --memory_size ${uex} \
                                --model ${model} \
                                --seed $SEED \
                                --n_tokens ${N_TOKENS} \
                                --adapt_type ${adapt_type} \
                                --online \
                                --lr ${lr} \
                                --log_prefix ${LOG_PREFIX}_${SEED}_model_${model}_lr${lr}_memsize${uex}_uex${uex}_from_${dataset2}_to_${dataset1} \
                                --load_checkpoint_path ${load_checkpoint_path} \
                              2>&1 | tee raw_logs/from_${dataset2}_to_${dataset1}_${LOG_PREFIX}_${SEED}_job${i}.txt &
                wait_n
                i=$((i + 1))
              done
            fi
          done
        done
      done
    done
  done
done
