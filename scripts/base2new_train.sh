#!/bin/bash

cd ..

# custom config
DATA=/home/svosve/Music/ma/ResPro/DATA
TRAINER=ResidualPrompting
DATASET=$1
CFG=vit_b16_ep50_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=4  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

MLP=LSTM2
# MLP=transformer
RESIDUAL=True
SEPARATE=True

for DATASET in 'dtd' 'eurosat' 'oxford_pets' 'oxford_flowers' 'caltech101' 'fgvc_aircraft' 'stanford_cars'
do
    for SEED in 1 2 3
    do
        DIR=/home/svosve/Music/ma/ResPro/output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}_${MLP}_${RESIDUAL}_${SEPARATE}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python /home/svosve/Music/ma/ResPro/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file /home/svosve/Music/ma/ResPro/configs/datasets/${DATASET}.yaml \
            --config-file /home/svosve/Music/ma/ResPro/configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
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

    for SEED in 1 2 3
    do
        COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}_${MLP}_${RESIDUAL}_${SEPARATE}/${CFG}/seed${SEED}
        MODEL_DIR=/home/svosve/Music/ma/ResPro/output/base2new/train_base/${COMMON_DIR}
        DIR=/home/svosve/Music/ma/ResPro/output/base2new/test_${SUB}/${COMMON_DIR}

        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python /home/svosve/Music/ma/ResPro/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file /home/svosve/Music/ma/ResPro/configs/datasets/${DATASET}.yaml \
            --config-file /home/svosve/Music/ma/ResPro/configs/trainers/${TRAINER}/${CFG}.yaml \
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
done
