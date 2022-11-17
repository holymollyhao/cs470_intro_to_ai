LOG_PREFIX="221114_normalized_src" ## after normalization
METHODS="ttaprompttune"

#INITIAL parameters for running model
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0
DATASET="sst-2" #this is fixed for source training

for METHOD in $METHODS; do
  if [ "${METHOD}" = "Src" ]; then
    EPOCH=1
    MODEL="bert"
    TGT="test"
    python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
                    --method ${METHOD} \
                    --tgt $TGT \
                    --model $MODEL \
                    --epoch $EPOCH \
                    --seed $SEED \
                    --log_prefix ${LOG_PREFIX}_${SEED} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

  elif [ "${METHOD}" = "prompttune" ]; then
    ITER_NUM=30000
    MODEL="bert"
    TGT="test"
    python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
                    --lr 0.3 \
                    --method ${METHOD} \
                    --tgt $TGT \
                    --model $MODEL \
                    --iter_step $ITER_NUM \
                    --seed $SEED \
                    --log_prefix ${LOG_PREFIX}_${SEED} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

  elif [ "${METHOD}" = "ttaprompttune" ]; then
    ITER_NUM=30000
    MODEL="bert"
    TGT="test"
    python main.py  --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
                    --lr 0.3 \
                    --method ${METHOD} \
                    --tgt $TGT \
                    --model $MODEL \
                    --iter_step $ITER_NUM \
                    --seed $SEED \
                    --log_prefix ${LOG_PREFIX}_${SEED} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
    fi
  i=$((i + 1))
done
