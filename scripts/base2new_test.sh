#!/bin/bash

cd ..

# custom config
USERPATH=/userpath
DATA=${USERPATH}/DATA
TRAINER=PRE
DATASET=$1
CFG=vit_b16_ep50_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

ENCODER=LSTM1
RESIDUAL=True
SEPARATE=False

LOADEP=50
SUB=new

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}_${ENCODER}_${RESIDUAL}_${SEPARATE}/${CFG}/seed${SEED}
    MODEL_DIR=${USERPATH}/output/base2new/train_base/${COMMON_DIR}
    DIR=${USERPATH}/output/base2new/test_${SUB}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python ${USERPATH}/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file ${USERPATH}/configs/datasets/${DATASET}.yaml \
        --config-file ${USERPATH}/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.PRE.ENCODER ${ENCODER} \
        TRAINER.PRE.RESIDUAL ${RESIDUAL} \
        TRAINER.PRE.SEPARATE ${SEPARATE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
