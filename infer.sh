while getopts ":d:s:" opt
do
    case $opt in
        d)
        DATASET="$OPTARG"
        ;;
        s)
        DECODING_STRATEGY="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/src
DATA_DIR=${PROJECT_PATH}/data/finetune/${DATASET}
SAVE_DIR=${DATA_DIR}/checkpoints

echo '-------- inference on dataset: '"$DATASET"'--------'

if [ "$DECODING_STRATEGY" == "greedy" ]; then
  echo '-------- decoding strategy: greedy --------'
  # inference
  BEAM=1
  LENPEN=1
  CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
  OUTPUT_FILE=${DATA_DIR}/output.txt
  PRED_FILE=${DATA_DIR}/pred.txt
  TASK=ved_translate
  fairseq-generate "${DATA_DIR}"/binary \
    --path "${CHECK_POINT}" \
    --user-dir ${USER_DIR} \
    --task ${TASK} \
    --batch-size 64 \
    --gen-subset test \
    --beam ${BEAM} \
    --num-workers 4 \
    --no-repeat-ngram-size 3 \
    --lenpen ${LENPEN} \
    2>&1 >"${OUTPUT_FILE}"
  grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
elif [ "$DECODING_STRATEGY" == "beam" ]; then
  echo '-------- decoding strategy: beam search --------'
  # inference
  BEAM=5
  LENPEN=1
  CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
  OUTPUT_FILE=${DATA_DIR}/output.txt
  PRED_FILE=${DATA_DIR}/pred.txt
  TASK=ved_translate
  fairseq-generate "${DATA_DIR}"/binary \
    --path "${CHECK_POINT}" \
    --user-dir ${USER_DIR} \
    --task ${TASK} \
    --batch-size 64 \
    --gen-subset test \
    --beam ${BEAM} \
    --num-workers 4 \
    --no-repeat-ngram-size 3 \
    --lenpen ${LENPEN} \
    2>&1 >"${OUTPUT_FILE}"
  grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
elif [ "$DECODING_STRATEGY" == "sampling"  ]; then
  echo '-------- decoding strategy: sampling --------'
  LENPEN=1
  TOP_K=30
  TOP_P=0.9
  TEM=0.65
  CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
  OUTPUT_FILE=${DATA_DIR}/output.txt
  PRED_FILE=${DATA_DIR}/pred.txt
  TASK=ved_translate
  fairseq-generate "${DATA_DIR}"/binary \
  --path "${CHECK_POINT}" \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --batch-size 64 \
  --gen-subset test \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen ${LENPEN} \
  --sampling \
  --sampling-topk ${TOP_K} \
  --sampling-topp ${TOP_P} \
  --temperature ${TEM} \
  --nbest 1 \
  --beam 1 \
  2>&1 >"${OUTPUT_FILE}"  # 将标准输出（stdout）和标准错误（stderr）都重定向到 ${OUTPUT_FILE} 文件中
  grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
else
  echo 'decoding strategy '"$DECODING_STRATEGY"' not found!'
  exit 1
fi
