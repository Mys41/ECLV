import os
import itertools

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    indexed_dataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from .utils import BertDictionary
from .dataset import LanguagePairDatasetVED


# 用于检查指定的数据集文件是否存在。它首先使用os.path.join函数和format方法构造数据集文件的文件名。文件名的格式为
# split.src-tgt.lang，其中 split、src、tgt和lang分别对应函数的前四个参数
def split_exists(split, src, tgt, lang, data_path, dataset_impl):
    filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
    return indexed_dataset.dataset_exists(filename, impl=dataset_impl)


# 加载语言对数据集
def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, add_cls_to_source=False,
        mask_source=False, masked_prob=0.15, masked_span_len=2, min_masked_len=15,
        auto_infer_absolute_positions=False, auto_infer_relative_positions=False,
):
    src_datasets = []  # 源序列数据集
    tgt_datasets = []  # 目标序列数据集

    # itertools.count()返回一个迭代器，该迭代器会生成一个无限递增的整数序列。所以每次循环中，k的值都会递增1
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # 使用split_exists函数检查源序列的数据集文件是否存在。如果存在，则构造数据集文件的前缀，
        # 否则检查目标序列的数据集文件是否存在。如果两个数据集文件都不存在，则根据k的值决定是抛出异常还是退出循环
        if split_exists(split_k, src, tgt, src, data_path, dataset_impl):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path, dataset_impl):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        # 使用data_utils.load_indexed_dataset函数加载源序列的数据集
        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        # 如果truncate_source=True，则对源序列数据集进行截断、去除尾部标记和添加尾部标记等操作
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        # 加载目标序列的数据集
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)  # 保证源序列与目标序列的数据集长度一致

    # 如果列表中只有一个数据集，那么直接将这个数据集赋值给src_dataset和tgt_dataset变量
    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        # 否则，构造一个长度为len(src_datasets)的列表sample_ratios，其中所有元素都为1。
        sample_ratios = [1] * len(src_datasets)
        # 然后，将列表中的第一个元素设置为upsample_primary。
        sample_ratios[0] = upsample_primary
        # 接下来，使用ConcatDataset类构造两个新的数据集。这个类接受两个参数：一个数据集列表和一个采样比例列表。
        # 它会根据指定的采样比例从数据集列表中的每个数据集中采样元素，并将它们拼接在一起，形成一个新的数据集。
        # 因为sample_ratios除了第一个元素都是1，所以会从每个数据集中等概率采样元素
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    # 在源序列和目标序列数据集的开头添加BOS标记
    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    # 加载对齐数据集
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    return LanguagePairDatasetVED(
        src_dataset, src_dataset.sizes, src_dict,  # 源序列数据集，大小，字典
        tgt_dataset, tgt_dataset.sizes, tgt_dict,  # 目标序列数据集，大小，字典
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,  # 最大源序列位置
        max_target_positions=max_target_positions,  # 最大目标序列位置
        remove_eos_from_source=False,
        append_eos_to_target=False,
        append_bos=False,
        align_dataset=align_dataset,  # 对齐数据集
        add_cls_to_source=add_cls_to_source,
        mask_source=mask_source,
        masked_prob=masked_prob,
        masked_span_len=masked_span_len,
        min_masked_len=min_masked_len,
        auto_infer_absolute_positions=auto_infer_absolute_positions,
        auto_infer_relative_positions=auto_infer_relative_positions,
    )


# @register_task('ved_translate')装饰器用于将DialogVEDTask类注册为一个新的任务类型，名称为'ved_translate'
# 等价于DialogVEDTask = register_task('ved_translate')(DialogVEDTask)
@register_task('ved_translate')
class DialogVEDTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args

    # 静态方法使用@staticmethod装饰器定义，它不接受隐式的第一个参数。也就是说，它的参数列表与普通函数相同，
    # 不需要额外添加self或cls参数。静态方法通常用于定义与类相关但不依赖于类或实例状态的方法
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)  # 确保在添加新的命令行参数之前，先添加父类定义的命令行参数

        parser.add_argument('--add-cls-to-source', default=False, action='store_true',
                            help='whether to add [CLS] token to the begin of sentence or not, '
                                 'it\'s recommended to include in VAE-based models')
        parser.add_argument('--mask-source', default=False, action='store_true', help='whether to mask input or not')
        parser.add_argument('--masked-prob', type=float, default=0.15, help='masked probability')
        parser.add_argument('--masked-span-len', type=int, default=2, help='masked span length')
        parser.add_argument('--min-masked-len', type=int, default=15, help='minimal source length if masked')
        parser.add_argument('--tokens-per-sample', type=int, default=512, help='masked probability')
        parser.add_argument('--auto-infer-absolute-positions', default=True, action='store_true',
                            help='whether to auto infer absolute positions')
        parser.add_argument('--auto-infer-relative-positions', default=False, action='store_true',
                            help='whether to auto infer relative positions')

    # 类方法使用@classmethod装饰器定义，它接受一个隐式的第一个参数cls。这个参数表示调用类方法的类本身。
    # 类方法通常用于定义与类相关且依赖于类状态的方法
    @classmethod
    def load_dictionary(cls, vocab_path: str):  # 加载BERT词典
        return BertDictionary.build_dictionary(vocab_path=vocab_path, has_freq=False)

    @classmethod
    def setup_task(cls, args, **kwargs):  # 设置任务，并返回一个新的任务对象

        paths = args.data.split(':')
        assert len(paths) > 0

        # 自动查找语言对
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        d = cls.load_dictionary(vocab_path='/home/taoran/ECLV/vocab.txt')
        print('| dictionary: {} types'.format(len(d)))

        return cls(args, d, d)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):  # 加载数据集
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            add_cls_to_source=self.args.add_cls_to_source,
            mask_source=self.args.mask_source,
            masked_prob=self.args.masked_prob,
            masked_span_len=self.args.masked_span_len,
            auto_infer_absolute_positions=self.args.auto_infer_absolute_positions,
            auto_infer_relative_positions=self.args.auto_infer_relative_positions,
        )

    def max_positions(self):  # 返回源序列和目标序列的最大位置
        return self.args.max_source_positions, self.args.max_target_positions


@register_task('translation_prophetnet')
class DialogVEDTaskPure(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, vocab_path: str):
        return BertDictionary.build_dictionary('vocab.txt', False)

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions
