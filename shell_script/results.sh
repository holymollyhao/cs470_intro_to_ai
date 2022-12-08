LOG_PREFIX="submission_results" ## after normalization

#INITIAL parameters for running model
############# run in single GPU ##############
GPUS=(0)
NUM_GPUS=1
##############################################


# if you want to run command on multiple GPUS,
# uncomment below line and replace upper code
##############################################
# GPUS=(0 1 2 4 5 6 7)
# NUM_GPUS=${#GPUS[@]}
##############################################

i=0


sleep 1 # prevent mistake
mkdir raw_logs # save console outputs here

#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=1 #num concurrent jobs
  local num_max_jobs=1
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

N_TOKENS=10
EPOCH=1
DATASETS="finefood imdb sst-2"
LR="0.001"
METHODS="ttaprompttune ln_tent Src"
UEX="32"
MODELS="distilbert"
SEEDS="0"
for SEED in $SEEDS; do
  for model in $MODELS; do
    for method in $METHODS; do
      for dataset1 in $DATASETS; do
        for dataset2 in $DATASETS; do
          for lr in $LR; do
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
                              2>&1 | tee raw_logs/${LOG_PREFIX}_${SEED}_model_${model}_from_${dataset2}_to_${dataset1}}_job${i}.txt &
                wait_n
                i=$((i + 1))
              else
                if [ "${method}" = "ttaprompttune" ]; then
                    ADAPT_TYPE="embed"
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
                                      2>&1 | tee raw_logs/${LOG_PREFIX}_${SEED}_model_${model}_lr${lr}_memsize${uex}_uex${uex}_from_${dataset2}_to_${dataset1}_job${i}.txt &
                      wait_n
                      i=$((i + 1))
                    done
                elif [ "${method}" = "ln_tent" ]; then
                    ADAPT_TYPE="ln"
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
                                    2>&1 | tee raw_logs/${LOG_PREFIX}_${SEED}_model_${model}_lr${lr}_memsize${uex}_uex${uex}_from_${dataset2}_to_${dataset1}_job${i}.txt &
                      wait_n
                      i=$((i + 1))
                    done
                fi
              fi
            done
          done
        done
      done
      wait
    done
  done
done

