LOG_PREFIX="submission_results" ## after normalization

# GPUS=(0 1 2 3 4 5 6 7)
GPUS=(0)
NUM_GPUS=${#GPUS[@]}
i=0


wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=1 #12
  local num_max_jobs=${1:-$default_num_jobs}
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

N_TOKENS=30
EPOCH=1
DATASETS="finefood imdb sst-2"
LR="0.01 0.00001"
METHODS="Src ln_tent dattaprompttune"
UEX="16 32 128"
MODELS="bert distilbert"
SEED="0 1 2"


for seed in $SEED; do
  for model in $MODELS; do
    for dataset1 in $DATASETS; do
      for dataset2 in $DATASETS; do
        for lr in $LR; do
          for method in $METHODS; do
            for uex in $UEX; do
              if [ "${model}" = "distilbert" ]; then
                load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/submission_source_${seed}_epoch1_lr0.00005_modeldistilbert/cp/cp_last.pth.tar
              elif [ "${model}" = "bert" ]; then
                load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/submission_source_${seed}_epoch1_lr0.00005_modelbert/cp/cp_last.pth.tar
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
                                --seed $seed \
                                --n_tokens ${N_TOKENS} \
                                --log_prefix ${LOG_PREFIX}_${seed}_model_${model}_from_${dataset2}_to_${dataset1} \
                                --load_checkpoint_path ${load_checkpoint_path} \
                              2>&1 | tee raw_logs/${LOG_PREFIX}_${seed}_model_${model}_from_${dataset2}_to_${dataset1}_job${i}.txt &
                wait_n
                i=$((i + 1))
              else
                if [ "${method}" = "ttaprompttune" ]; then
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
                                  --seed $seed \
                                  --n_tokens ${N_TOKENS} \
                                  --adapt_type ${adapt_type} \
                                  --online \
                                  --lr ${lr} \
                                  —log_prefix ${LOG_PREFIX}_${seed}_model_${model}_lr${lr}_memsize${uex}_uex${uex}_from_${dataset2}_to_${dataset1} \
                                  —load_checkpoint_path ${load_checkpoint_path} \
                                2>&1 | tee raw_logs/${LOG_PREFIX}_${seed}_model_${model}_lr${lr}_memsize${uex}_uex${uex}_from_${dataset2}_to_${dataset1}.txt &
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
done


#                          —load_checkpoint_path /home/twkim/git/tetra/log/sst-2/Src/tgt_test/221114_initial_gen_0/cp/cp_last.pth.tar \

# LOG_PREFIX="221125_results" ## after normalization

# #INITIAL parameters for running model
# #GPUS=(1 2 3 4 5 6 7)
# GPUS=(0)
# NUM_GPUS=${#GPUS[@]}
# SEED=0
# i=0


# wait_n() {
#   #limit the max number of jobs as NUM_MAX_JOB and wait
#   background=($(jobs -p))
#   local default_num_jobs=1 #12
#   local num_max_jobs=1
#   echo $num_max_jobs
#   if ((${#background[@]} >= num_max_jobs)); then
#     wait -n
#   fi
# }

# # hyperparameters, datasets, models
# N_TOKENS=30
# EPOCH=1
# METHODS="Src ttaprompttune ln_tent"
# DATASETS="finefood imdb sst-2"
# LR="0.01 0.00001"
# UEX="16 32 128"
# MODELS="bert distilbert"

# # best configuration for reproduction
# declare -A best_config
# declare -A DATASETS1
# declare -A MODELS
# DATASETS1[0]="finefood"
# DATASETS1[1]="imdb"
# DATASETS1[2]="sst-2"

# MODELS[0]="bert"
# MODELS[1]="distilbert"

# best_config[0,0,0]="16"
# best_config[0,0,1]="0.01"
# best_config[0,1,0]="32"
# best_config[0,1,1]="0.01"
# best_config[1,0,0]="16"
# best_config[1,0,0]="0.00001"
# best_config[1,1,1]="32"
# best_config[1,1,1]="0.00001"
# best_config[2,0,0]="128"
# best_config[2,0,1]="0.00001"
# best_config[2,1,0]="128"
# best_config[2,1,1]="0.00001"
# index1="0 1 2"
# index2="0 1"

# for id1 in $index1; do
#   for id2 in $index2; do
#     dataset1=${DATASETS1[$id1]}
#     model=${MODELS[$id2]}
#     uex=${best_config[$id1,$id2,0]}
#     lr=${best_config[$id1,$id2,1]}
    
#     for dataset2 in $DATASETS; do
#       for method in $METHODS; do
#         if [ "${model}" = "distilbert" ]; then
#           load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/221125_source_0_epoch1_lr0.00002_modeldistilbert/cp/cp_last.pth.tar
#         elif [ "${model}" = "bert" ]; then
#           load_checkpoint_path=log/${dataset2}/Src/src_train/tgt_test/221125_source_0_epoch1_lr0.00002_modelbert/cp/cp_last.pth.tar
#         fi

#         if [ "${method}" = "Src" ]; then
#           python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
#                           --dataset "${dataset1}" \
#                           --method "${method}" \
#                           --src train \
#                           --tgt test \
#                           --epoch 0 \
#                           --update_every_x ${uex} \
#                           --memory_size ${uex} \
#                           --model ${model} \
#                           --seed $SEED \
#                           --n_tokens ${N_TOKENS} \
#                           --log_prefix ${LOG_PREFIX}_${SEED}_model_${model}_from_${dataset2}_to_${dataset1} \
#                           --load_checkpoint_path ${load_checkpoint_path} \
#                         2>&1 | tee raw_logs/from_${dataset2}_to_${dataset1}_${LOG_PREFIX}_${SEED}_job${i}.txt &
#           wait_n
#           i=$((i + 1))
#         else
#           if [ "${method}" = "ttaprompttune" ]; then
#               ADAPT_TYPE="all all_ln_bn"
#           elif [ "${method}" = "ln_tent" ]; then
#               ADAPT_TYPE="ln"
#           fi

#           for adapt_type in $ADAPT_TYPE; do
#             python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} \
#                             --dataset "${dataset1}" \
#                             --method "${method}" \
#                             --src train \
#                             --tgt test \
#                             --epoch 1 \
#                             --update_every_x ${uex} \
#                             --memory_size ${uex} \
#                             --model ${model} \
#                             --seed $SEED \
#                             --n_tokens ${N_TOKENS} \
#                             --adapt_type ${adapt_type} \
#                             --online \
#                             --lr ${lr} \
#                             --log_prefix ${LOG_PREFIX}_${SEED}_model_${model}_lr${lr}_memsize${uex}_uex${uex}_from_${dataset2}_to_${dataset1} \
#                             --load_checkpoint_path ${load_checkpoint_path} \
#                           2>&1 | tee raw_logs/from_${dataset2}_to_${dataset1}_${LOG_PREFIX}_${SEED}_job${i}.txt &
#             wait_n
#             i=$((i + 1))
#           done
#         fi
#       done
#     done
#   done
# done




# #                          --load_checkpoint_path /home/twkim/git/tetra/log/sst-2/Src/tgt_test/221114_initial_gen_0/cp/cp_last.pth.tar \

