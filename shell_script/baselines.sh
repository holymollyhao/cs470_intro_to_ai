LOG_PREFIX="221116_baseline_ablation_tests" ## after normalization
METHODS="ln_tent"
METHOD="ln_tent"

#INITIAL parameters for running model
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
SEED=0
i=0
DATASET="finefood" #this is fixed for source training
#ADAPT_TYPE="all bn ln bnln emb"
LRs="0.3 0.2 0.1 0.01 0.001"
for lr in $LRs; do
  MODEL="bert"
  SRC="test"
  TGT="train"
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
                  --log_prefix ${LOG_PREFIX}_${SEED}_lr${lr}_memsize${memsize}_uex${uex}_type \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
  i=$((i + 1))
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