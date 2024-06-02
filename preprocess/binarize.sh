# this script is used pre-process fine-tune dialogue corpus esconv
# running this script will cause all files with the same name in the `BINARY_DIR` folder will be overwritten

PROJECT_PATH=/home/taoran/ECLV

USER_DIR=${PROJECT_PATH}/src
VOCAB_PATH=${PROJECT_PATH}/vocab.txt
NUM_WORKERS=20

# pre-process(esconv)
########################################################################################################################
DATA_DIR=${PROJECT_PATH}/data/finetune/esconv
PROCESSED_DIR=${DATA_DIR}/processed
BINARY_DIR=${DATA_DIR}/binary

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}
