#!/bin/bash

cd ..

# custom config
DATA=/home/ducan/Downloads/MinhAnh/CoOp/DATA
TRAINER=ResidualPrompting
# TRAINER=CoOp
# TRAINER=KgCoOp
WEIGHT=$2
DATASET=$1
#CFG=rn50_ep100  # config file\
CFG=vit_b16_ep50_ctxv1
# CFG=rn50_ep50
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
N=4  # number of proxy

for SEED in 2
do
    DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    #     echo "Run this job and save the output to ${DIR}"
    #     python train.py \
    #     --root ${DATA} \
    #     --seed ${SEED} \
    #     --trainer ${TRAINER} \
    #     --dataset-config-file configs/datasets/${DATASET}.yaml \
    #     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    #     --output-dir ${DIR} \
    #     TRAINER.COOP.N_CTX ${NCTX} \
    #     TRAINER.COOP.CSC ${CSC} \
    #     TRAINER.COOP.W ${WEIGHT} \
    #     TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    #     DATASET.NUM_SHOTS ${SHOTS} \
    #     DATASET.SUBSAMPLE_CLASSES base
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=50
SUB=new

for SEED in 2
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    DIR=output/base2new/test_${SUB}/${COMMON_DIR}


    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
