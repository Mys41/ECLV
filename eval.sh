while getopts ":d:" opt
do
    case $opt in
        d)
        DATASET="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

echo '-------- evaluate on dataset: '"$DATASET"'--------'

PROJECT_PATH='.'
DATA_DIR=${PROJECT_PATH}/data/finetune/${DATASET}
PRED_FILE=${DATA_DIR}/pred.txt
SAVE_DIR=${DATA_DIR}/checkpoints
CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
USER_DIR=${PROJECT_PATH}/src
TASK=ved_translate

python utils/evaluate.py \
  -name esconv \
  -hyp "${PRED_FILE}" \
  -ref "${DATA_DIR}"/processed/test.tgt
