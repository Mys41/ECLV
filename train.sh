while getopts ":p:t:d:" opt
do
    # 使用了getopts命令来解析命令行选项，选项可以是-p，-t或-d。如果选项是-p，则将其参数值赋给变量PRETRAINED_MODEL_PATH；
    # 如果选项是-t，则将其参数值赋给变量PRETRAINED_MODEL_TYPE；如果选项是-d，则将其参数值赋给变量DATASET。
    # 如果遇到未知的选项，则打印“未知参数”并退出程序。
    case $opt in
        p)
        PRETRAINED_MODEL_PATH="$OPTARG"
        ;;
        t)
        PRETRAINED_MODEL_TYPE="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

PROJECT_PATH='.'

# 检查变量PRETRAINED_MODEL_TYPE的值并根据其值设置变量ARCH的值
if [ "$PRETRAINED_MODEL_TYPE" == "dialogved_standard" ]; then
  echo '-------- model type: dialogved standard --------'
  ARCH=ngram_transformer_prophet_vae_standard
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large" ]; then
  echo '-------- model type: dialogved large --------'
  ARCH=ngram_transformer_prophet_vae_large
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_seq2seq"  ]; then
  echo '-------- model type: dialogved seq2seq --------'
  ARCH=ngram_transformer_prophet_seq2seq
else
  echo 'model type '"$PRETRAINED_MODEL_TYPE"' not found!'
  exit 1
fi

echo '-------- fine-tune on dataset: esconv --------'
NUM_WORKERS=6
CRITERION=ved_loss
TASK=ved_translate
USER_DIR=${PROJECT_PATH}/src
DATA_DIR=${PROJECT_PATH}/data/finetune/esconv
SAVE_DIR=${DATA_DIR}/checkpoints
TB_LOGDIR=${DATA_DIR}/tensorboard
fairseq-train \
  ${DATA_DIR}/binary \
  --fp16 \
  --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 1.0 \
  --lr 3e-5 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 190 \
  --criterion $CRITERION --label-smoothing 0.1 \
  --update-freq 4 --max-tokens 4500 --max-sentences 8 \
  --num-workers ${NUM_WORKERS}  \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.0 --weight-decay 0.01 \
  --encoder-layer-drop 0.0 \
  --save-dir ${SAVE_DIR} \
  --max-epoch 6 \
  --keep-last-epochs 10 \
  --no-last-checkpoints \
  --best-checkpoint-metric ppl \
  --max-source-positions 512 \
  --max-target-positions 128 \
  --kl-loss-weight 0.0 \
  --target-kl 5.0 \
  --strategy-rc-loss-weight 1.0 \
  --cls-bow-loss-weight 0.5 \
  --latent-bow-loss-weight 1.0 \
  --masked-lm-loss-weight 0.0 \
  --tensorboard-logdir ${TB_LOGDIR} \
  --dataset-impl mmap \
  --empty-cache-freq 64 \
  --seed 1 \
  --skip-invalid-size-inputs-valid-test \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
