#!/bin/bash

cd ..

# custom config
DATA=/home/ducan/Downloads/MinhAnh/CoOp/DATA
TRAINER=PLOT 

DATASET=$1
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
N=4  # number of proxy

# for SHOTS in 1 2 4 8 16
# do
for SEED in 1
do
DIR=/home/ducan/Downloads/MinhAnh/PLOT/plot-coop/output/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    python /home/ducan/Downloads/MinhAnh/PLOT/plot-coop/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file /home/ducan/Downloads/MinhAnh/PLOT/plot-coop/configs/datasets/${DATASET}.yaml \
    --config-file /home/ducan/Downloads/MinhAnh/PLOT/plot-coop/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.PLOT.N_CTX ${NCTX} \
    TRAINER.PLOT.CSC ${CSC} \
    TRAINER.PLOT.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.PLOT.N ${N} 
fi
done
# done