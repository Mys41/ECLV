import torch
import os
import numpy
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertTokenizer, RobertaTokenizer
from torch.nn.utils.rnn import pad_sequence
import re

sys.path.append("/home/taoran/ECLV/utils")

from comet_atomic2020_bart.utils import use_task_specific_params, trim_batch
from models.emotion_cause_model import KBCIN
import argparse, math
from tqdm import tqdm, trange
import json
import transformers

# 设置transformers库的日志级别为错误(error)，这样就只有错误信息会被打印出来，警告信息会被忽略
transformers.logging.set_verbosity_error()


# 检查f_path是否存在，如果存在则删除它，否则创建它所在的目录。其中exist_ok=True表示如果目录已经存在也不会报错。
def remove_or_makedir(f_path: str) -> None:
    if os.path.exists(f_path):
        os.remove(f_path)
    else:
        # make dir if not exit
        os.makedirs(os.path.dirname(f_path), exist_ok=True)


# 根据path的类型创建目录。如果path是一个字符串，则使用os.makedirs函数创建目录。
# 如果path是一个列表，则遍历列表中的每个元素，并对每个元素递归调用make_dir函数。
def make_dir(path):
    if isinstance(path, str):
        os.makedirs(path, exist_ok=True)
    elif isinstance(path, list):
        for p in path:
            make_dir(p)
    else:
        raise ValueError


# 对策略进行正则化
def norm_strategy(strategy):
    norm_str = "-".join(strategy.split())
    return "@[" + norm_str + "]"


# 获得策略
def get_strategy(file_path, norm=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d.lower().replace('[', '').replace(']', '') for d in data]
    if norm:
        data = [norm_strategy(d) for d in data]

    return data


# fin是一个字符串，表示原始数据集文件的路径；src_fout和tgt_fout都是可选的字符串参数，分别表示模型的源(输入)文件和目标(输出)文件的输出路径
# 首先，创建一个Roberta分词器对象tok。然后，使用open函数以只读模式打开输入文件，并使用readlines方法读取所有行到列表fin中
def esconv_prepare(fin: str, src_fout: str = None, tgt_fout: str = None) -> tuple:
    tok = BertTokenizer.from_pretrained('bert-base-uncased')  # "bert-base"
    strategy = get_strategy('../data/finetune/esconv/original_data/strategy.json', norm=True)
    strategy_list = [v for k, v in enumerate(strategy)]
    print("strategy_list:", strategy_list)
    # 使用的特殊标记
    special_tokens = ["[emotion type]", "[problem type]", "[situation]", "[emotion cause]"]
    tok.add_tokens(strategy_list)  # 将策略加入到词表中
    tok.add_tokens(special_tokens)  # 将特殊标记加入到词表中
    with open(fin, 'r', encoding='utf-8') as f:  # 读取字典
        my_dict_list = json.load(f)
    fin = my_dict_list
    # 如果指定了src_fout参数，则调用remove_or_makedir函数确保输出路径存在，然后使用open函数以写入模式打开源文件
    if src_fout:
        remove_or_makedir(src_fout)
        src_fout = open(src_fout, 'w', encoding='utf-8')
    # 如果制定了tgt_fout参数，也一样操作
    if tgt_fout:
        remove_or_makedir(tgt_fout)
        tgt_fout = open(tgt_fout, 'w', encoding='utf-8')
    return tok, fin, src_fout, tgt_fout


# 使用split方法将line按照__eou__分割成多个句子，然后对每个句子使用strip方法去除首尾空格，
# 再使用BERT分词器的tokenize方法进行分词。最后，使用' '.join(tokens)将分词结果连接成字符串，并返回一个列表。
def split_line_base(line: str, tok: BertTokenizer) -> list:
    return [' '.join(tokens) for tokens in [
        tok.tokenize(sent.strip()) for sent in line.split('__eou__')]]


# 对语句进行简单的预处理
def _norm(x):
    return ' '.join(x.strip().split())


# 获得常识知识
def get_csk_feature(model, tokenizer, context, relations):
    map1 = [{}]
    for id, conv in context.items():  # 获得上下文
        list1 = [[]]
        for utterance in conv:  # 遍历上下文中的语句
            queries = []  # 创建一个空列表，表示查询
            # 使用字符串拼接来生成一个查询字符串，并将查询字符串添加到列表queries中
            for r in relations:
                queries.append("{} {} [GEN]".format(utterance, r))
            with torch.no_grad():
                # 使用comet的分词器来对queries列表进行编码，并将结果转换为PyTorch张量
                batch = tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to('cuda')
                # 获得ids和注意力掩码
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)
                # 调用comet模型来计算输出，并将结果赋值给变量out
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                activations = out['decoder_hidden_states'][-1][:, 0, :].detach().cpu().numpy()
                for k, l1 in enumerate(list1):
                    l1.append(activations[k])
        # 将列表list1中对应位置的列表添加到当前字典中
        for k, v1 in enumerate(map1):
            v1[id] = list1[k]
    return map1


# 获得发言者掩码
def get_speaker_mask(totalIds, speakers):
    intra_masks, inter_masks = {}, {}
    for i in trange(len(totalIds)):
        id = totalIds[i]
        cur_speaker_list = speakers[id]  # 获得当前的发言者列表
        cur_intra_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        target_speaker = cur_speaker_list[-1]  # 目标语句的发言者
        target_index = len(cur_speaker_list) - 1  # 目标语句的索引

        cur_intra_mask[target_index][target_index] = 1
        for j in range(len(cur_speaker_list) - 1):
            # 如果该元素等于目标发言人，则将内部掩码数组中目标索引和当前索引对应的位置设为1。
            # 否则，将外部掩码数组中目标索引和当前索引对应的位置设为1。
            if cur_speaker_list[j] == target_speaker:
                cur_intra_mask[target_index][j] = 1
            else:
                cur_inter_mask[target_index][j] = 1

            # 全连接
            if j == 0:
                cur_intra_mask[j][j] = 1
            else:
                for k in range(j):
                    if cur_speaker_list[j] == target_speaker:
                        cur_intra_mask[j][k] = 1
                    else:
                        cur_inter_mask[j][k] = 1

        intra_masks[id] = cur_intra_mask
        inter_masks[id] = cur_inter_mask

    return intra_masks, inter_masks


# 获得相对位置
def get_relative_position(totalIds, speakers):
    relative_position = {}
    thr = 31
    for i in trange(len(totalIds)):
        id = totalIds[i]
        cur_speaker_list = speakers[id]
        cur_relative_position = []
        target_index = len(cur_speaker_list) - 1
        # 如果目标索引减去当前索引小于31，则将目标索引减去当前索引添加到列表cur_relative_position中
        # 否则，将31添加到列表cur_relative_position中
        for j in range(len(cur_speaker_list)):
            if target_index - j < thr:
                cur_relative_position.append(target_index - j)
            else:
                cur_relative_position.append(31)
        relative_position[id] = cur_relative_position
    return relative_position


def get_emotion_cause(ids, target_context, speaker, emotion, cause_labels):
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=4e-5, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--accumulate_step', type=int, required=False, default=1)
    parser.add_argument('--weight_decay', type=float, required=False, default=3e-4)
    parser.add_argument('--scheduler', type=str, required=False, default='constant')
    parser.add_argument('--warmup_rate', type=float, required=False, default=0.06)
    parser.add_argument('--rec-dropout', type=float, default=0.3, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--mlp_dropout', type=float, default=0.07, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--speaker_num', type=int, default=9, metavar='SN', help='number of speakers')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--num_attention_heads', type=int, default=6, help='Number of output mlp layers.')
    parser.add_argument('--hidden_dim', type=int, default=300, metavar='HD', help='hidden feature dim')
    parser.add_argument('--emotion_dim', type=int, default=300, metavar='HD', help='hidden feature dim')
    parser.add_argument('--roberta_dim', type=int, default=1024, metavar='HD', help='hidden feature dim')
    parser.add_argument('--csk_dim', type=int, default=1024, metavar='HD', help='hidden feature dim')
    parser.add_argument('--seed', type=int, default=42, metavar='seed', help='seed')
    parser.add_argument('--norm', action='store_true', default=False, help='normalization strategy')
    parser.add_argument('--save', action='store_true', default=True, help='whether to save best model')
    parser.add_argument('--add_emotion', action='store_true', default=True, help='whether to use emotion info')
    parser.add_argument('--use_emo_csk', action='store_true', default=True,
                        help='whether to use emo commonsense knowledge')
    parser.add_argument('--use_act_csk', action='store_true', default=True,
                        help='whether to use act commonsense knowledge')
    parser.add_argument('--use_event_csk', action='store_true', default=True, help='whether to use event knowledge')
    parser.add_argument('--use_pos', action='store_true', default=True, help='whether to use position embedding')
    parser.add_argument('--rnn_type', default='GRU', help='RNN Type')
    parser.add_argument('--model_size', default='base', help='roberta-base or large')
    parser.add_argument('--model_type', type=str, required=False, default='v2')
    parser.add_argument('--conv_encoder', type=str, required=False, default='none')
    parser.add_argument('--rnn_dropout', type=float, required=False, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")

    args = parser.parse_args()
    # print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    # if args.cuda:
    #     print('Running on GPU')
    # else:
    #     print('Running on CPU')
    cuda = args.cuda

    # 情感类型为8种
    if args.add_emotion:
        emotion_num = 8  # 7 categories plus 1 padding
    else:
        emotion_num = 0

    model = KBCIN(args, emotion_num)  # KBCIN模型

    seed = 1
    # 加载状态字典
    state_dict = torch.load('../KBCIN/save_dicts/best_model_{}'.format(str(seed)) + '.pkl')

    # 将状态字典加载到模型中
    model.load_state_dict(state_dict, strict=False)

    # 遍历模型中的所有参数
    for n, p in model.named_parameters():
        # 对于每个参数，判断其是否需要计算梯度。如果需要梯度，则打印出参数的名称和大小
        if p.requires_grad:
            # print(n, p.size())
            # 判断参数的形状是否大于1。如果大于1，则使用xavier_uniform_函数对参数进行初始化；
            # 否则，计算标准差stdv，并使用uniform_函数对参数进行初始化
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    if cuda:
        model.cuda()

    token_ids, attention_mask = {}, {}
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # "roberta-base"
    did = 1

    # print("Tokenizing Input Dialogs ...")  # 对输入数据进行分词
    for id, v in target_context.items():  # 遍历对话上下文
        cur_token_ids, cur_attention_mask = [], []
        for utterance in v:
            utterance = re.sub(r'@\[[^\]]*\]', '', utterance).strip()
            encoded_output = tokenizer(utterance)  # 对语句进行分词
            tid = encoded_output.input_ids  # 获得id表示
            atm = encoded_output.attention_mask  # 获得掩码，即忽略pad0
            cur_token_ids.append(torch.tensor(tid, dtype=torch.long))  # 将当前语句的id加入id列表中
            cur_attention_mask.append(torch.tensor(atm))
        tk_id = pad_sequence(cur_token_ids, batch_first=True, padding_value=1)  # 将句子pad至长度一致
        at_mk = pad_sequence(cur_attention_mask, batch_first=True, padding_value=0)
        token_ids[id] = tk_id  # 将id加入字典
        attention_mask[id] = at_mk

    # print("Generating Speaker Connections ...")
    intra_mask, inter_mask = get_speaker_mask(ids, speaker)  # 获得发言者掩码

    # print("Generating Relative Position ...")
    relative_position = get_relative_position(ids, speaker)  # 获得相对位置

    qmask = torch.FloatTensor([[1, 0] if x == 'A' else [0, 1] for x in speaker[str(did)]])
    umask = torch.FloatTensor([1] * len(cause_labels))

    model_path = "../KBCIN/comet_atomic2020_bart/comet-atomic_2020_BART/"
    comet_tokenizer = AutoTokenizer.from_pretrained(model_path)  # 加载comet的分词器
    comet = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()  # 加载comet
    use_task_specific_params(comet, "summarization")

    bf_event = ["isBefore"]  # 事件关系
    af_event = ["isAfter"]  # 事件关系
    xW_social = ["xWant"]
    xR_social = ["xReact"]
    oW_social = ["oWant"]
    oR_social = ["oReact"]

    # 获得常识知识特征
    bf = get_csk_feature(comet, comet_tokenizer, target_context, bf_event)
    bf = numpy.array(bf[0][str(did)])
    bf = torch.FloatTensor(bf)
    af = get_csk_feature(comet, comet_tokenizer, target_context, af_event)
    af = numpy.array(af[0][str(did)])
    af = torch.FloatTensor(af)
    xW = get_csk_feature(comet, comet_tokenizer, target_context, xW_social)
    xW = numpy.array(xW[0][str(did)])
    xW = torch.FloatTensor(xW)
    xR = get_csk_feature(comet, comet_tokenizer, target_context, xR_social)
    xR = numpy.array(xR[0][str(did)])
    xR = torch.FloatTensor(xR)
    oW = get_csk_feature(comet, comet_tokenizer, target_context, oW_social)
    oW = numpy.array(oW[0][str(did)])
    oW = torch.FloatTensor(oW)
    oR = get_csk_feature(comet, comet_tokenizer, target_context, oR_social)
    oR = numpy.array(oR[0][str(did)])
    oR = torch.FloatTensor(oR)

    #  提取相对位置字典中的元素并转换为张量
    relative_position = torch.tensor(relative_position[str(did)]).to('cuda')
    emotion_label = torch.tensor(emotion[str(did)]).to('cuda')
    token_ids_ = token_ids[str(did)].to('cuda')
    attention_mask_ = attention_mask[str(did)].to('cuda')
    token_ids = []
    attention_mask = []
    token_ids.append(token_ids_)
    attention_mask.append(attention_mask_)
    intra_mask = torch.from_numpy(intra_mask[str(did)]).to('cuda')
    inter_mask = torch.from_numpy(inter_mask[str(did)]).to('cuda')
    bf = bf.to('cuda')
    af = af.to('cuda')
    xW = xW.to('cuda')
    xR = xR.to('cuda')
    oW = oW.to('cuda')
    oR = oR.to('cuda')

    # print("token_ids:", token_ids)
    # print("attention_mask:", attention_mask)
    # print("emotion_label size:", emotion_label.size())
    # print("emotion_label:", emotion_label)
    # print("relative_position size:", relative_position.size())
    # print("relative_position:", relative_position)
    # print("intra_mask size:", intra_mask.size())
    # print("intra_mask:", intra_mask)
    # print("inter_mask size:", inter_mask.size())
    # print("inter_mask:", inter_mask)
    log_prob, _, _ = model(token_ids, attention_mask, emotion_label + 1, relative_position, intra_mask,
                           inter_mask, bf, af, xW, xR, oW, oR, qmask, umask)
    lp_ = log_prob.view(-1)  # [batch*seq_len]
    pred_ = torch.gt(lp_.data, 0.5).long()  # [batch*seq_len]
    return pred_


def split_line(
        line: str,
        tok: BertTokenizer,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = False,
        use_knowledge: bool = False) -> tuple:
    if not has_knowledge:
        assert not use_knowledge
    # 如果使用知识，则使用split方法将line按照制表符分割成三部分：知识、源文本和目标文本。
    # 然后，使用之前定义的split_line_base函数对知识和源文本进行分割，并使用knowledge_sep.join和sep.join方法
    # 将分割结果连接成字符串。最后，使用connect_sep + sep.join(split_line_base(src, tok))将知识和源文本连接起来。
    if has_knowledge and use_knowledge:
        knowledge, src, tgt = line.strip().split('\t')
        src_line = knowledge_sep.join(split_line_base(knowledge, tok)) + connect_sep + sep.join(
            split_line_base(src, tok))
    else:
        if has_knowledge and not use_knowledge:
            _, src, tgt = line.strip().split('\t')
        else:
            src, tgt = line.strip().split('\t')
        src_line = sep.join(split_line_base(src, tok))
    tgt_line = sep.join(split_line_base(tgt, tok))
    return src_line, tgt_line


# 处理ESConv数据集中的每个字典
def esconv_split_line(
        line: dict,
        tok: BertTokenizer,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = False,
        use_knowledge: bool = False,
        length: int = 6) -> tuple:
    if not has_knowledge:
        assert not use_knowledge

    # 如果使用知识，则使用split方法将line按照制表符分割成三部分：知识、源文本和目标文本。
    # 然后，使用之前定义的split_line_base函数对知识和源文本进行分割，并使用knowledge_sep.join和sep.join方法
    # 将分割结果连接成字符串。最后，使用connect_sep + sep.join(split_line_base(src, tok))将知识和源文本连接起来。
    if has_knowledge and use_knowledge:
        # 数据集中所有的情感类型组成的字典
        emotion_dict = {'anxiety': 0, 'depression': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'disgust': 5, 'shame': 6}
        emotion = {}
        target_context = {}
        speaker = {}
        ids = []

        history = []
        knowledge = []  # 加入对话历史的知识(用户状态)
        dialog = line['dialog']
        emotion_type = line['emotion_type']
        knowledge.append("[emotion type]" + line['emotion_type'])
        knowledge.append("[problem type]" + line['problem_type'])
        knowledge.append("[situation]" + line['situation'])
        tmp_speaker = 'supporter'

        context = []
        speaker_list = []
        emo = []
        cause_labels = []
        cur_emo = emotion_type  # 获取当前情感
        # 在训练集中，有一部分注释的情感需要修改
        if cur_emo == 'nervousness':
            cur_emo = 'anxiety'
        if cur_emo == 'pain':
            cur_emo = 'sadness'
        if cur_emo == 'jealousy':
            cur_emo = 'anger'
        if cur_emo == 'guilt':
            cur_emo = 'shame'
        for index, tmp_dic in enumerate(dialog):
            # turn = index + 1  # 获取对话轮次
            # cause_labels = [0] * (length - 1)
            # if turn == length:
            #     for i in range(turn):
            #         print("in")
            #         emo.append(emotion_dict[cur_emo])  # 将当前情感加入情感列表
            #         speaker_list.append(dialog[i]['speaker'])  # 将当前发言者加入发言者列表
            #         context.append(dialog[i]['content'])  # 将当前语句加入上下文列表

            if index == 0 and tmp_dic['speaker'] != 'seeker':
                continue
            if tmp_dic['speaker'] == tmp_speaker:
                continue
            tmp_speaker = tmp_dic['speaker']
            # 如果是用户的话语
            if tmp_dic['speaker'] == 'seeker':
                text = _norm(tmp_dic['content'])  # 获得对话文本
                history.append(text)  # 将用户语句添加到对话历史
                emo.append(emotion_dict[cur_emo])  # 将当前情感加入情感列表
                speaker_list.append(tmp_dic['speaker'])  # 将当前发言者加入发言者列表
                context.append(text)  # 将当前语句加入上下文列表
            # 如果是系统的话语
            if tmp_dic['speaker'] != 'seeker':
                # 在每个系统语句之前加上策略
                text = norm_strategy(tmp_dic['annotation']['strategy']) + _norm(tmp_dic['content'])
                history.append(text)
                emo.append(emotion_dict[cur_emo])  # 将当前情感加入情感列表
                speaker_list.append(tmp_dic['speaker'])  # 将当前发言者加入发言者列表
                context.append(text)  # 将当前语句加入上下文列表
            if len(history) == length:
                break

        # print("emo:", emo)
        # print("speaker_list:", speaker_list)
        did = 1
        ids.append(str(did))
        target_context[str(did)] = context[:-1]  # 将除了回答语句外的上下文加入到字典
        cause_labels = [0] * (len(context) - 1)
        speaker[str(did)] = speaker_list[:-1]
        emotion[str(did)] = emo[:-1]
        if length > 2:
            # 使用KBCIN模型预测情绪原因语句的索引
            pred = get_emotion_cause(ids, target_context, speaker, emotion, cause_labels)
            # 获取pred张量中非零元素的索引
            indices = torch.nonzero(pred).squeeze()
            # 获取偶数索引
            even_indices = indices[indices % 2 == 0]
            # 获取history列表中对应元素
            if even_indices.dim() > 0:
                emotion_cause = [history[i] for i in even_indices]
                knowledge.append("[emotion cause]" + ' '.join(emotion_cause))
        # print("token_ids:", token_ids)
        # print("attention_mask:", attention_mask)
        # print("emotion_label size:", emotion_label.size())
        # print("emotion_label:", emotion_label)
        # print("relative_position size:", relative_position.size())
        # print("relative_position:", relative_position)
        # print("intra_mask size:", intra_mask.size())
        # print("intra_mask:", intra_mask)
        # print("inter_mask size:", inter_mask.size())
        # print("inter_mask:", inter_mask)

        knowledge = '__eou__'.join(knowledge)
        src = '__eou__'.join(history[:-1])
        tgt = history[-1]
        src_line = knowledge_sep.join(split_line_base(knowledge, tok)) + connect_sep + sep.join(
            split_line_base(src, tok))
    else:
        history = []  # 一轮对话的对话历史
        if has_knowledge and not use_knowledge:
            # _, src, tgt = line.strip().split('\t')
            dialog = line['dialog']
            tmp_speaker = 'supporter'
            for index, tmp_dic in enumerate(dialog):
                if index == 0 and tmp_dic['speaker'] != 'seeker':
                    continue
                if tmp_dic['speaker'] == tmp_speaker:
                    continue
                tmp_speaker = tmp_dic['speaker']
                text = _norm(tmp_dic['content'])  # 获得对话文本
                history.append(text)
                if len(history) == length:
                    break
            src = '__eou__'.join(history[:-1])
            tgt = history[-1]
        else:
            # src, tgt = line.strip().split('\t')
            dialog = line['dialog']
            tmp_speaker = 'supporter'
            for index, tmp_dic in enumerate(dialog):
                if index == 0 and tmp_dic['speaker'] != 'seeker':
                    continue
                if tmp_dic['speaker'] == tmp_speaker:
                    continue
                tmp_speaker = tmp_dic['speaker']
                text = _norm(tmp_dic['content'])  # 获得对话文本
                history.append(text)
                if len(history) == length:
                    break
            src = '__eou__'.join(history[:-1])
            tgt = history[-1]
        src_line = sep.join(split_line_base(src, tok))
    tgt_line = sep.join(split_line_base(tgt, tok))
    return src_line, tgt_line


# 对之前定义的prepare和split_line函数进行单元测试
def unit_test():
    FINETUNE_PREFIX_PATH = "/home/taoran/ECLV/data/finetune"
    # 连接FINETUNE_PREFIX_PATH、esconv和original_data/dial.test
    fin = os.path.join(FINETUNE_PREFIX_PATH, 'esconv', 'original_data/dial.test')
    tok, fin, _, _ = esconv_prepare(fin)  # 调用prepare函数准备好分词器、模型需要的输入文件和输出文件
    # 然后，使用split_line函数对输入文件中的文本进行分割，并打印出分割结果
    src_line, tgt_line = esconv_split_line(fin[6], tok, has_knowledge=True, use_knowledge=True)
    print('source line: {}\ntarget line: {}\n'.format(src_line, tgt_line))
    src_line, tgt_line = esconv_split_line(fin[11], tok, has_knowledge=True, use_knowledge=True)
    print('source line: {}\ntarget line: {}\n'.format(src_line, tgt_line))


# 将原始数据集转换为模型需要的输入输出格式
def convert_esconv(
        fin: str,
        src_fout: str,
        tgt_fout: str,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = True,
        use_knowledge: bool = True,
        max_src_pos: int = 512,
        max_tgt_pos: int = 128,
        prune: bool = True) -> None:  # prune表示是否对过长的文本进行裁剪
    tok, fin, src_fout, tgt_fout = esconv_prepare(fin, src_fout, tgt_fout)  # 调用prepare函数准备好分词器、输入文件和输出文件
    num_prunes, num_src_prunes, num_tgt_prunes = 0, 0, 0  # 定义了三个计数器变量，用于统计被裁剪的行数、源文本和目标文本裁剪行数
    for line in tqdm(fin):  # 使用tqdm函数遍历输入文件中的每一个元素
        # 对话上下文长度为2，4，6，8
        for length in [2, 4, 6, 8]:
            src_line, tgt_lines = esconv_split_line(  # 调用split_line函数处理数据集中的每个字典，获得模型需要的输入输出
                line, tok, sep, knowledge_sep, connect_sep, has_knowledge, use_knowledge, length)
            tgt_lines = [ref.strip() for ref in tgt_lines.split('|')]
            src_line, tgt_line = src_line.split(), tgt_lines[0].split()
            # 如果源文本或目标文本的长度超过了最大长度，则更新裁剪计数
            if len(src_line) > max_src_pos or len(tgt_line) > max_tgt_pos:
                num_prunes += 1
                num_src_prunes += len(src_line) > max_src_pos
                num_tgt_prunes += len(tgt_line) > max_tgt_pos
            # 使用' '.join()将单词列表连接成字符串
            src_line = ' '.join(src_line[: max_src_pos - 1]) if prune else ' '.join(src_line)
            tgt_line = ' '.join(tgt_line[: max_tgt_pos - 1]) if prune else ' '.join(tgt_line)
            # 将转换后的源文本和目标文本写入到输出文件中
            src_fout.write('{}\n'.format(src_line))
            tgt_fout.write('{}\n'.format(tgt_line))
    src_fout.close()
    tgt_fout.close()
    # 打印出被裁剪的行数、源文本和目标文本裁剪行数
    print('{} lines exceed max positions, {} source lines and {} target lines have been pruned'.format(
        num_prunes, num_src_prunes, num_tgt_prunes))


# 检查处理后的文件中源文本和目标文本的最大长度
def check(processed_path: str, mode: str = 'test') -> None:
    # 使用open函数以只读模式打开源文件和目标文件，并使用readlines方法读取所有行到列表src_lines和tgt_lines中
    with open(os.path.join(processed_path, '{}.src'.format(mode)), encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(os.path.join(processed_path, '{}.tgt'.format(mode)), encoding='utf-8') as f:
        tgt_lines = f.readlines()
    # 定义了四个变量，用于记录源文本和目标文本的最大长度以及对应的目标文本和源文本的最大长度
    max_src, max_tgt, max_src_tgt, max_tgt_src = -1, -1, -1, -1
    # 使 zip函数遍历源文本和目标文本列表中的每一行，并使用split方法将每一行分割成单词列表。然后计算每一行的长度，并更新最大长度变量
    for src, tgt in zip(src_lines, tgt_lines):
        src_len = len(src.strip().split())
        tgt_len = len(tgt.strip().split())
        if src_len > max_src:
            max_src = src_len
            max_src_tgt = tgt_len
        if tgt_len > max_tgt:
            max_tgt = tgt_len
            max_tgt_src = src_len
    # 打印出处理后的文件路径、源文本和目标文本的最大长度以及对应的目标文本和源文本的最大长度
    print('{}\nboundary shape src: ({}, {})\nboundary shape tgt: ({}, {})\n'.format(
        processed_path, max_src, max_src_tgt, max_tgt_src, max_tgt))


if __name__ == '__main__':
    unit_test()
