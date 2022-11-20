LOG_PREFIX="221120_results" ## after normalization
METHODS="Src dattaprompttune ln_tent"

#INITIAL parameters for running model
GPUS=(1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
SEED=0
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
LR="0.3 0.1 0.01 0.001 0.0001 0.00001"
METHODS="dattaprompttune ln_tent"
UEX="16 32 64 128"
MODELS="bert distilbert"
for model in $MODELS; do
  for dataset1 in $DATASETS; do
    for dataset2 in $DATASETS; do
      for lr in $LR; do
        for method in $METHODS; do
          for uex in $UEX; do
            if [ "${model}" = "distilbert" ]; then
              load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/221119_source_backbone_true_0_epoch1_distilbert/cp/cp_last.pth.tar
            elif [ "${model}" = "bert" ]; then
#              /home/twkim/git/tetra/log/finefood/Src/tgt_test/221119_source_backbone_true_0_epoch1/cp
              load_checkpoint_path=log/${dataset2}/Src/tgt_test/221119_source_backbone_true_0_epoch1/cp/cp_last.pth.tar
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
              if [ "${method}" = "ttaprompttune" ]; then
                  ADAPT_TYPE="all all_ln_bn"
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



#                          --load_checkpoint_path /home/twkim/git/tetra/log/sst-2/Src/tgt_test/221114_initial_gen_0/cp/cp_last.pth.tar \

