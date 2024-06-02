import sys  # 添加包名的搜索路径
import os

sys.path.append("/home/taoran/ECLV")

# 现在可以导入父目录下的模块了
from utils.processor import convert_esconv, check

FINETUNE_PREFIX_PATH = '../data/finetune'

ORIGINAL_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'esconv/original_data')
PROCESSED_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'esconv/processed')

convert_esconv(
    fin=os.path.join(ORIGINAL_PATH, 'dial.train'),
    src_fout=os.path.join(PROCESSED_PATH, 'train.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'train.tgt'),
    use_knowledge=True
)
convert_esconv(
    fin=os.path.join(ORIGINAL_PATH, 'dial.valid'),
    src_fout=os.path.join(PROCESSED_PATH, 'valid.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'valid.tgt'),
    use_knowledge=True
)
convert_esconv(
    fin=os.path.join(ORIGINAL_PATH, 'dial.test'),
    src_fout=os.path.join(PROCESSED_PATH, 'test.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'test.tgt'),
    use_knowledge=True
)

check(PROCESSED_PATH, mode='train')
check(PROCESSED_PATH, mode='valid')
check(PROCESSED_PATH, mode='test')
