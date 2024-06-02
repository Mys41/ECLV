import torch
from fairseq.data import LanguagePairDataset
from fairseq.data import data_utils
import numpy as np

from .utils import (
    _infer_absolute_position_sentence_backward,
    _infer_absolute_position_role_backward_with_knowledge,
    _infer_relative_position_token,
    _infer_relative_position_sentence,
)


def collate(
        samples, pad_idx, sep_idx, soc_idx, cls_idx, eos_idx,
        left_pad_source=True, left_pad_target=False,
        input_feeding=True, mask_source=False,
        auto_infer_absolute_positions=False,
        auto_infer_relative_positions=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        # 我们对这个merge函数做了一个简单的更改，以适应masked的merge masked_indexs和masked_tokens
        # 构造了一个列表，其中每个元素都是samples列表中对应元素的key属性的值，如果该值为None，则使用一个空的张量代替
        # collate_tokens函数用于将多个张量拼接在一起，以形成一个批次。它接受几个参数，包括要拼接的张量列表、填充索引、
        # 结束符索引、是否左对齐和是否将结束符移动到开头。它会根据这些参数对输入的张量进行处理，然后将它们拼接在一起，形成一个批次
        return data_utils.collate_tokens(
            [s[key] if s[key] is not None else torch.empty(0) for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    # 检查对齐矩阵是否有效。它会检查矩阵中的索引是否在给定的源序列和目标序列的长度范围内。如果不在范围内，则认为对齐矩阵无效，并返回False
    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    # 计算对齐矩阵中每个目标位置的权重。它会计算每个目标位置在对齐矩阵中出现的次数，并将其倒数作为该位置的权重。
    # 这样，出现次数较多的目标位置将获得较小的权重，而出现次数较少的目标位置将获得较大的权重。
    def compute_alignment_weights(_alignments):
        align_tgt = _alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        _align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / _align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)  # 所有源序列标记

    # 计算每个样本的源序列长度，并根据长度对样本进行降序排序
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    # sort_order是一个一维张量，其长度与样本长度相同。它表示对samples列表中的元素进行降序排序后，每个元素在原列表中的索引。
    # 例如，如果samples列表中有3个元素，其源序列长度分别为3、1和2，则变量sort_order的值为：
    # tensor([0, 2, 1])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    # 检查samples列表中第一个元素的target属性是否为None。如果不为None，则执行以下操作：
    if samples[0].get('target', None) is not None:
        # 调用merge函数，将samples列表中每个元素的target属性合并在一起，并将结果存储在变量target中
        target = merge('target', left_pad=left_pad_target)
        # 使用PyTorch库中的index_select函数根据源序列长度对变量target进行重新排序
        target = target.index_select(0, sort_order)
        # 获得样本中每个元素的target长度列表，并转换为LongTensor，然后根据源序列长度行重新排序
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        # 计算样本中所有元素的target标记数量，并将结果存储在变量ntokens中
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # 我们创建目标序列的移位版本，用于将先前的输出标记馈送到下一个解码器步骤
            # 调用merge函数，将samples列表中每个元素的target属性合并在一起，并将结束符移动到开头。
            # 然后，将结果存储在变量prev_output_tokens中
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            # 然后根据源序列长度行重新排序
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    # 如果samples列表中第一个元素的target属性为None ，则执行以下操作：
    else:
        # 计算样本中所有元素的source标记数量，并将结果存储在变量ntokens中
        ntokens = sum(len(s['source']) for s in samples)

    # ------------------------------------------------------------------------------------------------------------------
    # 添加掩码索引和标记
    masked_tokens, masked_target = None, None

    if mask_source:
        # 由于src_token是按源序列长度排序的，如果第一个样本没有被masked，则意味着该批没有被屏蔽，因为最大句子长度仍然太短
        if samples[sort_order[0].item()].get('masked_tokens', None) is not None:
            masked_tokens = merge('masked_tokens', left_pad=left_pad_source)
            masked_target = merge('masked_target', left_pad=left_pad_source)
            masked_tokens = masked_tokens.index_select(0, sort_order)
            masked_target = masked_target.index_select(0, sort_order)

    # 计算masked标记总数
    n_masked_tokens = sum(len(
        s.get('masked_tokens')) if s.get('masked_tokens', None) is not None else 0 for s in samples)

    sentence_positions, role_positions = None, None

    # 计算绝对位置矩阵
    if auto_infer_absolute_positions:
        sentence_positions = _infer_absolute_position_sentence_backward(
            src_tokens, sep_idx, pad_idx, cls_idx)
        role_positions = _infer_absolute_position_role_backward_with_knowledge(
            src_tokens, sep_idx, pad_idx, soc_idx, cls_idx)

    relative_position_token, relative_position_sentence = None, None

    # 计算相对位置矩阵
    if auto_infer_relative_positions:
        relative_position_token = _infer_relative_position_token(src_tokens, pad_idx)
        relative_position_sentence = _infer_relative_position_sentence(src_tokens, sep_idx)

    # ------------------------------------------------------------------------------------------------------------------

    batch = {
        'id': id,
        'nsentences': len(samples),  # 样本长度
        'ntokens': ntokens,  # 样本中所有标记总数
        'n_masked_tokens': n_masked_tokens,  # 样本中masked标记总数
        'net_input': {
            'src_tokens': src_tokens,  # 所有源序列标记
            'src_lengths': src_lengths,  # 源序列长度
            'sentence_positions': sentence_positions,  # 句子绝对位置矩阵
            'role_positions': role_positions,  # 角色绝对位置矩阵
            'relative_position_token': relative_position_token,  # 标记级相对位置矩阵
            'relative_position_sentence': relative_position_sentence,  # 句子级相对位置矩阵
            'masked_tokens': masked_tokens,  # 所有被掩码的标记
        },
        'target': target,  # 目标序列标记
        'masked_target': masked_target,  # 掩码后的标记的原本标记
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # 检查samples列表中第一个元素的alignment属性是否为None。如果不为None，则执行以下操作：
    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape  # 获得批次大小和目标序列大小
        src_sz = batch['net_input']['src_tokens'].shape[1]  # 获得源序列大小

        # 构造一个形状为[len(sort_order), 2]的全零张量，len(sort_order)=样本长度
        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        # 将offsets第二列加上一个一维张量，其元素为(0, tgt_sz, 2*tgt_sz, ...)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        # 如果left_pad_source为真，则将offsets第一列加上要pad的长度
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        # 如果left_pad_target为真，则将offsets第二列加上要pad的长度
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        # 构造一个对齐矩阵列表，对齐矩阵大小为[len(sort_order), 2]。具体来说，它遍历对齐索引、偏移、源序列长度和目标序列长度，
        # 并从samples[align_idx]['alignment']中提取对齐矩阵。然后，它使用函数check_alignment检查对齐矩阵是否有效。
        # 如果有效，则将对齐矩阵加上偏移，并将结果添加到列表中。
        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        # 如果列表不为空，则将列表中所有对齐矩阵沿着第一个维度拼接在一起，并将结果存储在变量alignments中。
        # 然后，调用函数compute_alignment_weights计算对齐权重，并将结果存储在变量align_weights中
        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


# seq2seq/vae/rl预训练期间的对话语料库格式：
# context: <s> <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> </s>
# response: <s> <response> </s>

# *** Training ***
# encoder input: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
# <CLS>用于预测回答中的词袋
# encoder input: </s> <response>
# decoder output: <response> </s>
# </s>由于fairseq惯例而添加到开头，尽管我更喜欢`<s><response>`

# *** Inference ***
# encoder input: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
# first decode input: </s>
# 解码器将使用预测的标记作为新的输入，直到输出</s>

# 如果需要添加更多特殊符号，请在`vocab.txt`文件中将<unused_{}>替换为它
# 与LanguagePairDataset相比，LanguagePearDatasetVAE添加了一个参数add_cls_to_source
# 更多的定制化操作是可行的


class LanguagePairDatasetVED(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        append_bos=False,
        seed=1,
        # 新添加的参数
        add_cls_to_source=False, mask_source=False,
        masked_prob=0.15, masked_span_len=2, min_masked_len=15,
        auto_infer_absolute_positions=False,
        auto_infer_relative_positions=False,
    ):
        # add_cls_to_source：是否在每个示例的开头添加[CLS]标记
        # masked_source：是否masked源序列
        super().__init__(
            src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, left_pad_source, left_pad_target,
            max_source_positions, max_target_positions, shuffle, input_feeding,
            remove_eos_from_source, append_eos_to_target, align_dataset, append_bos
        )
        self.add_cls_to_source = add_cls_to_source
        self.mask_source = mask_source
        self.masked_prob = masked_prob
        self.masked_span_len = masked_span_len
        self.auto_infer_absolute_position = auto_infer_absolute_positions
        self.auto_infer_relative_positions = auto_infer_relative_positions

        # 如果输入太短，则不执行掩码
        self.min_masked_len = int(max(min_masked_len, int(1 / self.masked_prob * self.masked_span_len)))
        # replace_probs表示掩码的标记被替换为以下结果的概率：
        # 1.<MASK> 2.不变 3.字典中的随机标记
        self.replace_probs = torch.tensor([0.8, 0.1, 0.1])

    def __getitem__(self, index):
        # with data_utils.numpy_seed(self.seed, self.epoch, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # 如果没有EOS，则将EOS附加到tgt语句的末尾，如果有EOS，则从src语句的末尾删除EOS。当我们存在相反方向的数据集时，
        # 即当我们想将tgt_dataset用作src_dataset时，这是有用的，反之亦然
        if self.append_eos_to_target:  # 将EOS添加到tgt结尾
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:  # 将BOS添加到tgt和src开头
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:  # 将BOS从src删除
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        # clone them since there are some inplace operations below
        # 克隆它们，因为下面有一些就地操作
        src_item, tgt_item = src_item.clone(), tgt_item.clone()

        if self.add_cls_to_source:
            # 在每句话的开头加上[CLS]标记
            src_item = torch.cat([torch.LongTensor([self.src_dict.cls()]), src_item])

        source_len = len(src_item)  # 源序列长度
        masked_tokens, masked_target = None, None

        # 如果需要掩码源序列，并且源序列长度大于最小掩码长度
        if self.mask_source and source_len > self.min_masked_len:
            # 输出masked的标记和masked的序列
            # 我们不掩码[CLS]，因为：
            # 1）.[CLS]始终是第一个标记
            # 2）.[CLS]不同于其他标记，因为[CLS]需要理解整个上下文并连接到潜在空间
            masked_tokens = self.cal_mask_tokens(source_len)  # 要掩码的标记的索引
            masked_target = src_item[masked_tokens].clone()  # 要掩码的标记的原本标记
            # mask it
            src_item[masked_tokens] = self.replace(src_item[masked_tokens])  # 将要掩码的标记随机替换为三种mask

        example = {
            'id': index,
            'source': src_item,  # 掩码后的源序列
            'target': tgt_item,  # 目标序列
            'masked_tokens': masked_tokens,  # 要掩码的标记的索引
            'masked_target': masked_target,  # 掩码后的标记的原本标记
        }

        # 如果有对齐数据集，则添加到example字典中
        if self.align_dataset is not None and not isinstance(self.align_dataset, bool):
            print("align")
            example['alignment'] = self.align_dataset[index]
        return example

    # 随机选择要掩码的标记的索引
    def cal_mask_tokens(self, source_len):
        positions = np.arange(source_len)
        # 每个标记被选中的概率相同
        # masked_indice_start = positions[np.random.random(size=source_len)<self.masked_prob]
        # 掩码长度，输入越长，masked_len越长
        masked_len = min(int(round(self.masked_prob * source_len / self.masked_span_len)), 1)
        # 对特殊标记没有限制
        # 每个标记被屏蔽的概率相同
        # 一旦选择了一个标记，我们就扩展这个标记以形成一个跨度，并掩码整个跨度
        # np.random.choice函数从positions数组中随机选择masked_len个元素，且不放回。也就是说，每个元素只能被选择一次
        masked_tokens = np.random.choice(positions, masked_len, replace=False)
        # 扩展此标记以形成要预测的目标的跨度
        for step in range(1, self.masked_span_len):
            masked_tokens_end = masked_tokens + step
            masked_tokens = np.append(masked_tokens, masked_tokens_end)
        # 使用np.unique函数去除masked_tokens数组中的重复元素，并使用np.sort函数对数组进行排序。然后，它使用布尔索引保留
        # 小于source_len的元素
        masked_tokens = np.sort(np.unique(masked_tokens[masked_tokens < source_len]))
        masked_tokens = torch.tensor(masked_tokens, dtype=torch.int64)
        return masked_tokens  # 返回要掩码的标记的索引

    # 随机替换为三种掩码中的一种
    def replace(self, x):
        x_real = x.clone()  # 1.不变
        x_rand = x.clone().random_(self.src_dict.nspecial, len(self.src_dict))  # 2.字典中的随机标记
        x_mask = x.clone().fill_(self.src_dict.mask())  # 3.<MASK>
        # 这种采样需要一个类似datautils.torch_seed的函数来控制重复性
        # 使用torch.multinomial函数从self.replace_probs张量中随机采样len(x)个元素，且采样时放回
        probs = torch.multinomial(self.replace_probs, len(x), replacement=True)
        masked = torch.LongTensor(x_mask * (probs == 0) + x_real * (probs == 1) + x_rand * (probs == 2))
        return masked

    # 获得一个批次的数据
    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), sep_idx=self.src_dict.sep(),
            soc_idx=self.src_dict.soc(), cls_idx=self.src_dict.cls(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, mask_source=self.mask_source,
            auto_infer_absolute_positions=self.auto_infer_absolute_position,
            auto_infer_relative_positions=self.auto_infer_relative_positions,
        )
