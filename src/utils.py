import torch
import json
from fairseq.data import Dictionary
from fairseq.utils import make_positions

# 特殊标记id
# cls_id -> 4 | sep_id -> 5 | pad_id -> 1 | soc_id -> 9
PAD_INDEX = 0
KNOWLEDGE_ROLE = 3
MAX_SENTENCE = 31

"""
masked span预训练期间的对话表示：
<s> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> </s>

1. in a masked turn, a complete turn will be masked as [MASK1]
2. in unmasked turn, some tokens can be masked as [MASK2]
1.在一个掩码轮次中，完整轮次将被masked为[MASK1]
2.在未掩码的轮次中，一些标记可以被masked为[MASK2]：

seq2seq预训练期间的对话表示：
context: <s> <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> </s>
response: <s> <turn> </s>
如果需要添加更多特殊符号，请直接将其添加到vocab.txt中
"""


# 对策略进行正则化
def norm_strategy(strategy):
    norm_str = "-".join(strategy.split())
    return "@[" + norm_str + "]"


# 获得策略
def get_strategy(file_path, norm=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d.replace('[', '').replace(']', '') for d in data]
    if norm:
        data = [norm_strategy(d) for d in data]

    return data


# 这个字典模块贯穿了预处理、训练和推理的整个过程
class BertDictionary(Dictionary):
    def __init__(
            self,
            pad='[PAD]',
            eos='</s>',
            unk='[UNK]',
            bos='<s>'
    ):
        super().__init__(pad=pad, eos=eos, unk=unk, bos=bos)

        self.cls_word = '[CLS]'
        self.sep_word = '[SEP]'
        self.mask_word = '[MASK]'
        self.mask1_word = '[MASK1]'
        self.mask2_word = '[MASK2]'
        # 对话的开始，在此id之前的文本跨度被视为知识
        self.soc_word = '[SOC]'

    @classmethod
    def build_dictionary(cls, vocab_path: str, has_freq: bool):
        # bert dictionary from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        d = cls()
        # strategy = get_strategy('/nfs/home/taoran/DialogVED-main/data/finetune/'
        #                         'esconv/original_data/strategy.json', norm=True)
        # strategy_list = [v for k, v in enumerate(strategy)]
        # for token in strategy_list:
        #     d.add_symbol(token)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            # 如果has_freq为真，将读取文件中的每一行，并将每一行分割成两部分：单词和频率。然后，使用add_symbol方法，
            # 将单词添加到字典中，并指定其出现的次数。如果has_freq为假，将读取文件中的每一行，并将每一行作为一个单词
            if has_freq:
                # 每一行都是一个标记，其频率由空格分隔
                for line in f.readlines():
                    word, freq = line.strip().split()
                    d.add_symbol(word=word, n=freq)
            else:
                # 每一行都是一个标记
                for line in f.readlines():
                    word = line.strip()
                    d.add_symbol(word=word, n=1)
            d.add_symbol(word=word, n=1)
        d.nspecial = 999
        return d

    # 返回特殊标记
    def cls(self):
        return self.index(self.cls_word)

    def sep(self):
        return self.index(self.sep_word)

    def pad(self):
        assert self.index(self.pad_word) == self.pad_index
        return self.pad_index

    def mask(self):
        return self.index(self.mask_word)

    def mask1(self):
        return self.index(self.mask1_word)

    def mask2(self):
        return self.index(self.mask2_word)

    def soc(self):
        return self.index(self.mask2_word)


# 推断句子前向绝对位置
def _infer_absolute_position_sentence_forward(input_ids, sep_id, pad_id, cls_id):
    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0      2      0      3      0      4        5
    # 所有不属于一个特定轮次的标记都可以被视为pad

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    # 参考上面的上下文index，cumsum函数计算input_ids张量中每个元素等于sep_id的累积和，然后将结果加1，这样，
    # positions张量中的CLS和turn1就等于1，turn2等于2，以此类推
    positions = torch.cumsum(input_ids.eq(sep_id).int(), dim=1) + 1
    # 然后再将CLS等特殊标记的位置变为0
    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    # 防止数组下标超出索引
    positions[positions > MAX_SENTENCE] = MAX_SENTENCE

    return positions


# 推断句子后向绝对位置
def _infer_absolute_position_sentence_backward(input_ids, sep_id, pad_id, cls_id):
    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3>  <response>
    # index:       0      4      0      3      0      2         1
    # 所有不属于一个特定轮次的标记都可以被视为pad

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    # 与上面的位置相反
    positions = torch.cumsum(input_ids.flip(1).eq(sep_id).int(), dim=1).flip(1) + 2
    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    # 防止数组下标超出索引
    positions[positions > MAX_SENTENCE] = MAX_SENTENCE

    return positions


# 推断角色前向绝对位置
def _infer_absolute_position_role_forward(input_ids, sep_id, pad_id, cls_id):
    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0       2     0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    # 角色位置1，2，1，2循环
    positions = (torch.cumsum(input_ids.flip(1).eq(sep_id).int(), dim=1).flip(1) + 1) % 2 + 1

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX

    return positions


# 角色前向绝对位置与后向相同
def _infer_absolute_position_role_backward(input_ids, sep_id, pad_id, cls_id):
    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0       2     0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = torch.cumsum(input_ids.eq(sep_id).int(), dim=1) % 2

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX

    return positions


# 该函数是兼容的
# 推断带有知识的角色后向绝对位置
def _infer_absolute_position_role_backward_with_knowledge(input_ids, sep_id, pad_id, soc_id, cls_id):
    # 我们需要一个[SOT]标记来指示对话的开始
    # given context: <CLS> <know1> [SEP] <know2> [SOT] <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <know1> [SEP] <know2> [SOT] <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # output:      0      3      0      3      3      1      0      2      0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = (torch.cumsum(input_ids.flip(1).eq(sep_id), dim=1).flip(1) + 1) % 2 + 1
    # 添加KNOWLEDGE_ROLE，知识跨度应始终位于soc_id的左侧
    # [SOC]表示对话的开始
    alpha = torch.cumsum(input_ids.flip(1).eq(soc_id), dim=1).flip(1)
    positions = (1 - alpha) * positions + alpha * KNOWLEDGE_ROLE

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    positions[input_ids == soc_id] = PAD_INDEX

    return positions


def _infer_relative_position_token(input_ids, pad_id):
    """
    :param input_ids: (seq_len, batch_size)
    :param pad_id: <pad> index in the dictionary
    :return: token level relative position matrix before bucket
    """

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ]).transpose(1, 0)

    positions = make_positions(input_ids, padding_idx=pad_id).transpose(1, 0)

    # alpha = positions.eq(pad_id)
    # positions = (positions.unsqueeze(0) - positions.unsqueeze(1)).permute(2, 0, 1)
    # alpha = (alpha.unsqueeze(0) + alpha.unsqueeze(1) + 0).permute(2, 0, 1)
    # positions = (1 - alpha) * positions

    return (positions.unsqueeze(0) - positions.unsqueeze(1)).permute(2, 0, 1)


def _infer_relative_position_sentence(input_ids, sep_id):
    """
    a three-turns dialogue input sequence ids is supposed to be:
        <cls> <turn1> <sep> <turn2> <sep> <turn3>
    :param input_ids: (seq_len, batch_size)
    :param sep_id: <sep> index in the dictionary
    :return: turn level relative position matrix before bucket
    """

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ]).transpose(1, 0)

    positions = torch.cumsum(input_ids.transpose(1, 0).eq(sep_id).int(), dim=0)

    return (positions.unsqueeze(0) - positions.unsqueeze(1)).permute(2, 0, 1)
