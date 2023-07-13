#!/bin/bash

cd ..

# custom config
# DATA=/home/svosve/Music/ma/ResPro/DATA
DATA=/Users/miapham/Downloads/Master_Code/CoOp/DATA
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
MLP=MLP1
RESIDUAL=True
SEPARATE=True

for SEED in 1
do
    DIR=/Users/miapham/Documents/GitHub/ResPro/output/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}_${MLP}_${RESIDUAL}_${SEPARATE}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python /Users/miapham/Documents/GitHub/ResPro/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file /Users/miapham/Documents/GitHub/ResPro/configs/datasets/${DATASET}.yaml \
        --config-file /Users/miapham/Documents/GitHub/ResPro/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.ResidualPrompting.MLP ${MLP} \
        TRAINER.ResidualPrompting.RESIDUAL ${RESIDUAL} \
        TRAINER.ResidualPrompting.SEPARATE ${SEPARATE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=50
SUB=new

for SEED in 1
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}_${MLP}_${RESIDUAL}_${SEPARATE}/${CFG}/seed${SEED}
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
        TRAINER.ResidualPrompting.MLP ${MLP} \
        TRAINER.ResidualPrompting.RESIDUAL ${RESIDUAL} \
        TRAINER.ResidualPrompting.SEPARATE ${SEPARATE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
