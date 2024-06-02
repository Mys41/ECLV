import math
import random
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
    PositionalEmbedding
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .attention import (
    RelativePositionBias,
    MultiheadAttentionRPE,
    NgramMultiheadAttention,
    ngram_attention_bias,
    LearnedPositionalEmbeddingNew
)


# 在程序开始时读取词汇表
with open('/home/taoran/ECLV/vocab.txt', 'r') as f:
    vocab = [line.strip() for line in f]


# 预测未来ngram的TransformerModel
@register_model('ngram_transformer_prophet')
class NgramTransformerProphetModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        # --------------------------------------------------------------------------------------------------------------
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--add-bos-token', action='store_true', default=False)
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')

        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--encoder-layer-drop', type=float, default=0.0)

        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--ngram', type=int, metavar='N',
                            help='num of predicting grams')
        parser.add_argument('--num_buckets', type=int, metavar='N',
                            help='num of buckets for relative position')
        parser.add_argument('--relative_max_distance', type=int, metavar='N',
                            help='num of bucket for relative position')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings '
                                 '(requires shared dictionary and embed dim)')
        # --------------------------------------------------------------------------------------------------------------
        parser.add_argument('--with-mask-lm-logits', action='store_true',  # 是否包括掩码语言模型
                            help='whether to include masked language module')
        parser.add_argument('--with-cls-bow-logits', action='store_true',
                            help='whether to include [CLS] bag-of-word logits')
        parser.add_argument('--with-latent-bow-logits', action='store_true',
                            help='whether to include latent bag-of-word logits')
        parser.add_argument('--extend-latent-with-cls', action='store_true',  # 是否使用[CLS]特征扩展隐变量
                            help='whether to extend latent variable with [CLS] feature')

        parser.add_argument('--disable-kl-loss', action='store_true',
                            help='whether to disable kullback–leibler divergence loss')

        parser.add_argument('--with-encoder-ape-token', action='store_true')
        parser.add_argument('--with-encoder-ape-sentence', action='store_true')
        parser.add_argument('--with-encoder-ape-role', action='store_true')

        parser.add_argument('--with-encoder-rpe-token', action='store_true')
        parser.add_argument('--with-encoder-rpe-sentence', action='store_true')

        parser.add_argument('--load-from-pretrained-model', type=str, default=None,
                            help='Load from pretrained model')
        parser.add_argument('--deterministic', action='store_true', default=False,
                            help='whether to generate deterministic latent variable')

        parser.add_argument('--target-kl', default=3.0, type=float, help='target k-l loss')
        parser.add_argument('--kl-loss-weight', default=1.0, type=float, help='kl divergence loss weight')
        parser.add_argument('--strategy-rc-loss-weight', default=1.0, type=float, help='y reconstruct loss weight')
        parser.add_argument('--cls-bow-loss-weight', default=0.5, type=float, help='bag of word loss weight')
        parser.add_argument('--latent-bow-loss-weight', default=0.5, type=float, help='bag of word loss weight')
        parser.add_argument('--masked-lm-loss-weight', default=1.0, type=float, help='mask lm loss weight')

    # 概率归一化
    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # 如果解码器中有get_normalized_probs方法，则直接调用
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        # 如果net_output是一个张量，则将其转换为浮点类型并赋值给变量logits
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            # 根据log_probs，使用F.log_softmax或F.softmax函数对logits行归一化
            if log_probs:
                # F.log_softmax返回归一化后概率值的对数
                return F.log_softmax(logits, dim=-1)
            else:
                # softmax归一化会将输入张量中的每个元素转换为一个概率值，使得所有元素的和为1
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    @staticmethod
    def print_args(args):  # 输出参数
        # 定义了一个名为iargs的列表，其中包含了一些感兴趣的参数名称
        iargs = [
            # task specific
            'mask_source',
            'add_cls_to_source',
            'generate_latent_variable',
            # logits
            'with_mask_lm_logits',
            'with_cls_bow_logits',
            'with_latent_bow_logits',
            'extend_latent_with_cls',
            'use_latent_variable',
            'disable_kl_loss',
            # loss weight
            'masked_lm_loss_weight',
            'cls_bow_loss_weight',
            'latent_bow_loss_weight',
            'kl_loss_weight',
            # position embedding
            'with_encoder_ape_sentence',
            'with_encoder_ape_role',
            'with_encoder_rpe_token',
            'with_encoder_rpe_sentence',
            'deterministic',
            'latent_size',
        ]
        # 打印一行等号，用于分隔输出
        # ljust方法将字符串的长度扩展到66个字符
        print('='.ljust(66, '='))
        # 遍历args对象的所有属性，并将每个属性的名称存储在变量arg中
        for arg in vars(args):
            if arg in iargs:    
                print("{} = {}".format(arg, getattr(args, arg)))  # 打印属性的名称和值
        print('-'.ljust(66, '-'))  # 打印一行横线和当前时间戳
        print('| Time stamp: {}'.format(str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))))

    @staticmethod
    def check_args(args):  # 检查参数
        print('-'.ljust(66, '-'))
        # 掩码了语言跨度，但没有掩码语言损失
        if args.mask_source and not args.with_mask_lm_logits:
            warnings.warn(message='language span masked but with not masked language loss !')
        if args.with_mask_lm_logits:
            # assert args.mask_source
            # 计算了掩码语言损失，但损失权重不是正的
            if not args.masked_lm_loss_weight > 0.0:
                warnings.warn(message='masked lm logits computed but with not positive loss weight !')
        if args.generate_latent_variable:
            # 生成了隐变量，但词袋损失没有得到优化
            if not args.with_latent_bow_logits:
                warnings.warn(message='latent variable is generated but bag-of-word loss is not optimized !')
            # 隐变量由其他标记生成，而不是[CLS]
            if not args.add_cls_to_source:
                warnings.warn(message='latent variable is generated by other tokens but not [CLS] !')
            # 隐变量生成了但未使用
            if not args.use_latent_variable:
                warnings.warn(message='latent variable is generated but not used !')
        # cls词袋损失由其他标记生成，但不是[CLS]
        if args.with_cls_bow_logits and not args.add_cls_to_source:
            warnings.warn(message='cls bag-of-word logits is generated by other tokens but not [CLS] !')
        # 生成了cls词袋损失，但损失权重不是正的
        if args.with_cls_bow_logits and not args.cls_bow_loss_weight > 0.0:
            warnings.warn(message='cls bag-of-word logits is generated but with not positive loss weight !')
        # 隐变量由其他标记扩展，但不是[CLS]
        if args.extend_latent_with_cls and not args.add_cls_to_source:
            warnings.warn(message='latent variable is generated by other tokens but not [CLS] !')
        # 隐变量词袋损失由其他标记生成，但不是[CLS]
        if args.with_latent_bow_logits and not args.add_cls_to_source:
            warnings.warn(message='latent bag-of-word logits is generated by other tokens but not [CLS] !')
        # 生成了隐变量词袋损失，但损失权重不是正的
        if args.with_latent_bow_logits and not args.latent_bow_loss_weight > 0.0:
            warnings.warn(message='latent bag-of-word logits is generated but with not positive loss weight !')
        # 禁用k-l散度损失，但具有正的k-l损失权重
        if args.disable_kl_loss and args.kl_loss_weight > 0.0:
            warnings.warn(message='k-l divergence loss is disabled but with positive k-l loss weight !')
        if args.with_encoder_ape_sentence or args.with_encoder_ape_role:
            assert args.auto_infer_absolute_positions
        if args.with_encoder_rpe_token or args.with_encoder_rpe_sentence:
            assert args.auto_infer_relative_positions
        print('-'.ljust(66, '-'))

    @classmethod
    def build_model(cls, args, task):  # 建立模型

        # 加载默认模型参数/此函数可应用于fairseq中的所有模块
        # task/model/criterion etc.
        base_architecture(args)

        # check args
        # cls.check_args(args)

        # print args
        # cls.print_args(args)

        # 源序列词典和目标序列词典，在大多数翻译任务中，词典是不同的，但在典型的英汉文本到文本或数据到文本任务中，字典是一体的
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # 如果需要共享所有嵌入层
        if args.share_all_embeddings:
            # 共享嵌入层需要使用相同的字典
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            # 共享嵌入层需要编码器和解码器的嵌入维度相同
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            # 构建编码器的嵌入层，并将解码器的嵌入层设置为与编码器的嵌入层相同
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = encoder_embed_tokens
            # share parameters in bag_of_word predicted layer in VAE
        else:
            # 否则编码器和解码器各自构建嵌入层
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        # 如果共享所有嵌入层，则不会训练该模块的参数
        bow_embed_tokens_enc = nn.Linear(in_features=args.encoder_embed_dim, out_features=len(tgt_dict), bias=False)
        bow_embed_tokens_latent = nn.Linear(in_features=args.encoder_embed_dim, out_features=len(tgt_dict), bias=False)
        mask_embed_tokens = nn.Linear(in_features=args.encoder_embed_dim, out_features=len(tgt_dict), bias=False)

        # 初始编码器和解码器
        encoder = TransformerEncoder(  # Transformer编码器，加上了更复杂的位置嵌入
            args, src_dict, encoder_embed_tokens, bow_embed_tokens_enc, bow_embed_tokens_latent, mask_embed_tokens)
        decoder = NgramTransformerDecoder(args, tgt_dict, decoder_embed_tokens)  # 未来n个标记解码器

        model = NgramTransformerProphetModel(encoder, decoder)

        if args.load_from_pretrained_model is not None:
            print('loading pretrained model from {}'.format(args.load_from_pretrained_model))
            # torch.load函数从指定文件中加载预训练模型的状态。map_location='cpu'指定了将模型状态加载到CPU上
            states = torch.load(args.load_from_pretrained_model, map_location='cpu')
            if 'model' in states and 'args' in states:
                states = states['model']
            # replaced = {
            #     'encoder.embed_positions.weight': 'encoder.embed_ap_token.weight',
            #     'decoder.embed_positions.weight': 'decoder.embed_ap_token.weight',
            # }
            # for key in replaced:
            #     if key in states:
            #         _, _embed_dim = states[key].shape
            #         _dtype = states[key].dtype
            #         states[replaced.get(key)] = torch.cat(
            #             [torch.zeros(1, _embed_dim, dtype=_dtype), states[key]], dim=0)
            #         del states[key]
            # adapt to new positions setting
            # try:
            #     for position_name, target_position_length in [
            #         ('encoder.embed_ap_token.weight', model.encoder.embed_ap_token.weight.size(0)),
            #         ('decoder.embed_ap_token.weight', model.decoder.embed_ap_token.weight.size(0)),
            #     ]:
            #         if states[position_name].size(0) < target_position_length:
            #             _index = torch.arange(states[position_name].size(1))
            #             expend_position_states = states[position_name].clone()
            #             while states[position_name].size(0) < target_position_length:
            #                 _index = torch.cat((_index[1:], _index[:1]), dim=0)
            #                 states[position_name] = torch.cat([
            #                     states[position_name], expend_position_states[:, _index]], dim=0)
            #         if states[position_name].size(0) > target_position_length:
            #             states[position_name] = states[position_name][:target_position_length]
            # except (AttributeError, KeyError):
            #     pass
            # # delete unmatched keys
            # unmatched_keys = ['encoder.vae_fc3.weight', 'decoder.vae_transform.weight', 'decoder.vae_transform.bias']
            # for key in unmatched_keys:
            #     if key in states:
            #         del states[key]

            # 适应新的位置设置
            try:
                # 遍历编码器和解码器的位置嵌入权重。position_name存储了权重的名称，target_position_length存储了权重矩阵的目标大小
                for position_name, target_position_length in [
                    ('encoder.embed_positions_token.weight', model.encoder.embed_ap_token.weight.size(0)),
                    ('decoder.embed_positions_token.weight', model.decoder.embed_ap_token.weight.size(0)),
                ]:
                    # 判断权重矩阵的大小是否小于目标大小
                    if states[position_name].size(0) < target_position_length:
                        # 如果权重矩阵的大小小于目标大小，则创建一个索引张量，用于循环移位矩阵
                        _index = torch.arange(states[position_name].size(1))
                        # 创建一个权重矩阵的副本，用于扩展矩阵
                        expend_position_states = states[position_name].clone()
                        # 开始一个while循环，用于扩展权重矩阵，直到其大小达到目标大小
                        while states[position_name].size(0) < target_position_length:
                            # 在每次循环中，对索引张量进行循环移位
                            _index = torch.cat((_index[1:], _index[:1]), dim=0)
                            # 使用循环移位后的索引张量对权重矩阵进行扩展
                            states[position_name] = torch.cat([
                                states[position_name], expend_position_states[:, _index]], dim=0)
                    # 如果权重矩阵的大小大于目标大小，则将其截断为目标大小
                    if states[position_name].size(0) > target_position_length:
                        states[position_name] = states[position_name][:target_position_length]
            except (AttributeError, KeyError):
                pass
            # 加载预训练层
            model_dict = model.state_dict()
            # 与ProphetNet相比，更新了什么
            print('| new in current model: ')  # 现有模型中的新参数
            print([k for k, v in model_dict.items() if k not in states.keys()])
            print('| discard from original model: ')  # 从原始模型中丢弃的参数
            print([k for k, v in states.items() if k not in model_dict.keys()])
            state_dict = {k: v for k, v in states.items() if k in model_dict.keys()}
            print('| updating parameters ...')  # 更新模型参数
            print('-'.ljust(66, '-'))
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            args.load_from_pretrained_model = None

        return NgramTransformerProphetModel(encoder, decoder)

    def max_positions(self):
        return self.encoder.max_positions(), self.decoder.max_positions()

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, **kwargs):

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)

        return decoder_out, encoder_out


# 使用正态分布和零常数填充初始化torch.nn.Embedding
def Embedding(num_embeddings, embedding_dim, padding_idx):
    # num_embeddings参数表示嵌入层中嵌入向量的数量，即词汇表的大小
    # embedding_dim参数表示每个嵌入向量的维度
    # padding_idx参数表示填充索引。如果指定了这个参数，则在计算损失或梯度时，嵌入层会忽略输入中等于padding_idx的位置
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.normal_将嵌入层权重矩阵中的每个元素初始化为一个随机值，
    # 这些随机值服从均值为mean、标准差为std的正态分布。在这个例子中，均值为0，标准差为embedding_dim**-0.5
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    # 将权重矩阵的padding_idx行中的所有元素都设置为0
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


# 使用xavier均匀权重和零偏差初始化torch.nn.Linear
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


# 基于词典构建嵌入层
def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)  # 嵌入数量等于词典大小
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    return emb


def id_to_word(id_tensor, vocab):
    # 将ID张量转换为词
    words = [vocab[i] for i in id_tensor]

    return words


# Transformer编码器层
# 原始论文方法: dropout -> add residual -> layer_norm
# 残差连接的思想是将子层的输入与其输出相加，以便在反向传播过程中更好地传递梯度。这样，即使子层的输出梯度很小，
# 梯度仍然可以通过残差连接直接流向子层的输入，从而改善模型的训练效果
# multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
# position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
#
# tensor2tensor方法: layer_norm -> dropout -> add residual
# multi-head self-attention sublayer: x = x + dropout(self_attn(layer_norm(x))))
# position-wise feed-forward sublayer: x = x + dropout(fc2(dropout(relu(fc1(layer_norm(x))))))
# 在这篇论文中，我们采用了原始结构
class TransformerEncoderLayer(nn.Module):
    def __init__(
            self, encoder_embed_dim, encoder_ffn_embed_dim, encoder_attention_heads, dropout,
            attention_dropout, activation_dropout, activation_fn,
            embed_relative_positions_token=None, embed_relative_positions_sentence=None,
    ):
        super().__init__()
        self.embed_dim = encoder_embed_dim
        """
            与“MultiheadAttention”相比，“Multihead Attention RPE”配备了相对位置嵌入
            如果embed_relative_positions_token和embed_relative_positions_sentence均为None
            则该模块与MultiheadAttention完全相同
        """
        self.embed_relative_positions_token = embed_relative_positions_token  # 标记级相对位置
        self.embed_relative_positions_sentence = embed_relative_positions_sentence  # 句子级相对位置
        if embed_relative_positions_token is None and embed_relative_positions_sentence is None:
            self.self_attn = MultiheadAttention(
                self.embed_dim, encoder_attention_heads,
                dropout=attention_dropout, self_attention=True,
            )
        else:
            self.self_attn = MultiheadAttentionRPE(
                self.embed_dim, encoder_attention_heads,
                dropout=attention_dropout, self_attention=True,
                embed_relative_positions_token=embed_relative_positions_token,
                embed_relative_positions_sentence=embed_relative_positions_sentence,
            )
        # self.self_attn = MultiheadAttentionRPE(
        #     self.embed_dim, encoder_attention_heads,
        #     dropout=attention_dropout, self_attention=True,
        #     embed_relative_positions_token=embed_relative_positions_token,
        #     embed_relative_positions_sentence=embed_relative_positions_sentence,
        # )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask=None, relative_position_token=None, relative_position_sentence=None):
        """
        Args:
            x: input to the t-layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask: binary ByteTensor of shape `(batch, src_len)`
            relative_position_token: token relative position tensor of shape `(batch, src_len, src_len)`
            relative_position_sentence: token relative position tensor of shape `(batch, src_len, src_len)`
        """
        # Note:
        # `key_padding_mask`通常不是None，因为文本序列的长度可变
        # `attn_mask`通常为None，除非某些标记不需要注意其他标记
        # 但在解码器层中不是None

        # multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
        residual = x
        """
            注意：不同距离之间的标记应该有不同的注意力模式
            即使标记之间的距离相同，但句子之间的距离不同
            也可能存在不同的注意力模式
        """

        # x, _ = self.self_attn(
        #     query=x, key=x, value=x,
        #     key_padding_mask=encoder_padding_mask, need_weights=False,
        #     # relative position embedding (PE) here
        #     # relative token PE + relative sentence PE + relative token PE * relative sentence PE
        #     relative_position_token=relative_position_token,
        #     relative_position_sentence=relative_position_sentence,
        # )
        if self.embed_relative_positions_token is None and self.embed_relative_positions_sentence is None:
            x, _ = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=encoder_padding_mask, need_weights=False,
            )
        else:
            x, _ = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=encoder_padding_mask, need_weights=False,
                # relative position embedding (PE) here
                # relative token PE + relative sentence PE + relative token PE * relative sentence PE
                relative_position_token=relative_position_token,
                relative_position_sentence=relative_position_sentence,
            )
        # dropout -> add residual -> layer_norm
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        # position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
        residual = x
        x = self.activation_fn(self.fc1(x))
        # training指定了模型是否处于训练模式。如果为True，则会应用dropout；否则，不会应用dropout
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


# Transformer编码器
class TransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, bow_embed_tokens_enc=None,
                 bow_embed_tokens_latent=None, mask_embed_tokens=None):
        super().__init__(dictionary)

        self.dictionary = dictionary
        self.dropout = args.dropout
        self.encoder_layer_drop = args.encoder_layer_drop

        # 绝对位置嵌入设置
        self.with_encoder_ape_token = args.with_encoder_ape_token
        self.with_encoder_ape_sentence = args.with_encoder_ape_sentence
        self.with_encoder_ape_role = args.with_encoder_ape_role

        # 相对位置嵌入设置
        self.with_encoder_rpe_token = args.with_encoder_rpe_token
        self.with_encoder_rpe_sentence = args.with_encoder_rpe_sentence

        # 训练模式为seq2seq或vae
        self.generate_latent_variable = args.generate_latent_variable
        # 将[CLS]标记的隐藏状态映射到特征以预测词袋
        self.with_cls_bow_logits = args.with_cls_bow_logits
        # 将隐变量映射到特征以预测词袋
        self.with_latent_bow_logits = args.with_latent_bow_logits
        # 将掩码隐藏状态映射到特征以预测掩码语言
        self.with_mask_lm_logits = args.with_mask_lm_logits
        # 将隐变量与编码器特征连接起来预测词袋logits
        self.extend_latent_with_cls = args.extend_latent_with_cls

        # 变分自编码器设置
        # self.target_kl = args.target_kl if self.generate_latent_variable else None
        # self.deterministic = args.deterministic if not self.generate_latent_variable else None
        # self.disable_kl_loss = args.disable_kl_loss if not self.generate_latent_variable else None
        self.deterministic = getattr(args, 'deterministic', False)
        self.disable_kl_loss = getattr(args, 'disable_kl_loss', False)

        self.embed_dim = args.encoder_embed_dim
        # 全局args中的`encoder_embed_dim`参数必须等于self-attention中的参数'
        assert embed_tokens.embedding_dim == self.embed_dim, \
            '`encoder_embed_dim` parameter in global args must equal to the one in self-attention'

        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.share_all_embeddings = args.share_all_embeddings

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)

        # 绝对位置嵌入 (APE) -> cite PLATO here < https://arxiv.org/pdf/1910.07931.pdf >

        # 标记级位置嵌入 token PE (default padding index is 1)
        if self.with_encoder_ape_token:
            # args.max_source_positions_token表示嵌入层中嵌入向量的数量，即源序列的最大标记数量
            # learned=True表示使用可学习的位置嵌入
            self.embed_ap_token = PositionalEmbedding(
                args.max_source_positions_token, self.embed_dim, self.padding_idx, learned=True)
        else:
            self.embed_ap_token = None
            # 使用此设置存在风险，因为transformer模型对顺序不敏感！
            warnings.warn(message='there is a risk in using this setting because '
                                  'the transformer model is order insensitive !')

        # 句子级位置嵌入 sentence PE (padding index is 0)
        """
            回答句应该总是句子1，所以如果对话是：
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
            那么句子索引就是:
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
               5        4         3        2        1
            由于回答总是句子1，我们不需要在解码器中构建类似的模块
        """
        # 获得句子位置嵌入，args.max_source_positions_sentence表示源序列的最大句子数量
        self.embed_ap_sentence = PositionalEmbedding(
            args.max_source_positions_sentence, self.embed_dim, padding_idx=None, learned=True,
        ) if self.with_encoder_ape_sentence else None

        # 通过将位置嵌入层的第一个位置的权重设为0，可以使模型在处理序列的开始时忽略位置信息。
        # 这样，模型就可以根据输入数据中的特殊标记来识别序列的开始，而不是依赖位置信息
        if self.embed_ap_sentence is not None:
            nn.init.constant_(self.embed_ap_sentence.weight[0], 0)

        # 角色位置嵌入 role PE (padding index is 0)
        """
            回答角色应该始终是角色1，因此如果对话是：
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
            那么角色索引就是:
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
               1        2         1        2        1
            
            如果输入中有额外的知识，则应将其放在开头，并为额外的知识分配一个索引
            <knowledge 1> <knowledge 2> <knowledge 3> <turn 1> <turn 2> <turn 3> <turn 4> <response>
                 3              3            3           1        2        1         2         1
                 
            由于回答总是角色1，我们不需要在解码器中构建类似的模块
        """
        self.embed_ap_role = PositionalEmbedding(
            args.max_source_positions_role, self.embed_dim, padding_idx=None, learned=True,
        ) if self.with_encoder_ape_role else None

        # 将位置嵌入层的第一个位置的权重设为0
        if self.embed_ap_role is not None:
            nn.init.constant_(self.embed_ap_role.weight[0], 0)

        # 相对位置嵌入 (RPE) -> cite T5 & ProphetNet here
        self.embed_rp_token = RelativePositionBias(  # 相对位置偏置
            embed_dim=self.embed_dim,
            num_buckets=args.num_buckets_source_token,
            max_distance=args.max_distance_source_token,
            n_heads=args.encoder_attention_heads,
            bidirectional=args.bidirectional_source_token,
        ) if self.with_encoder_rpe_token else None

        self.embed_rp_sentence = RelativePositionBias(
            embed_dim=self.embed_dim,
            num_buckets=args.num_buckets_source_sentence,
            max_distance=args.max_distance_source_sentence,
            n_heads=args.encoder_attention_heads,
            bidirectional=args.bidirectional_source_sentence,
        ) if self.with_encoder_rpe_sentence else None

        self.layers = nn.ModuleList([])
        # embed_relative_positions层在所有编码器层中共享
        self.layers.extend([
            TransformerEncoderLayer(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                args.encoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
                self.embed_rp_token,
                self.embed_rp_sentence,
            )
            for _ in range(args.encoder_layers)
        ])

        # self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.layer_norm = LayerNorm(self.embed_dim)

        # 是否包括[CLS]词袋logits
        if self.with_cls_bow_logits:
            # 使用vae模块第一个线性层
            self.vae_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
            # 词袋预测器线性层
            self.bow_fc_enc = bow_embed_tokens_enc if not self.share_all_embeddings else None
        else:
            self.vae_fc1 = None

        # 如果生成隐变量
        if self.generate_latent_variable:
            # 如果没有编码器-解码器的注意力，它是变分自编码器模式
            if self.vae_fc1 is None:
                self.vae_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
            # 将vae_fc1特征映射到潜在均值和方差
            self.vae_fc2 = nn.Linear(self.embed_dim, 2 * args.latent_size)

            # 将隐变量映射到预测词袋的新特征
            if self.with_latent_bow_logits:  # 是否包括隐词袋logits
                if self.extend_latent_with_cls:  # 是否将[CLS]特征与隐变量连接起来
                    self.vae_fc3 = nn.Linear(self.embed_dim + args.latent_size, self.embed_dim)
                else:
                    self.vae_fc3 = nn.Linear(args.latent_size, self.embed_dim)
                    # warnings.warn(message='using this setting may cause the model hard to be trained !')
                self.bow_fc_latent = bow_embed_tokens_latent if not self.share_all_embeddings else None
            else:
                self.vae_fc3 = None
                self.bow_fc_latent = None
                # 使用此设置不使用隐变量的信息
                warnings.warn(message='using this setting do not utilize information the of latent variables!')

        # 用于掩码语言模型损失
        if self.with_mask_lm_logits:
            # 这种类型的线性层通常用于对输入数据进行线性变换，可以用作自注意力机制中的一部分，用于计算注意力权重
            self.mask_lm_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.mask_lm_fc2 = mask_embed_tokens if not self.share_all_embeddings else None

        self.apply(init_bert_params)  # 用于在使用fairseq库构建BERT模型时初始化模型的参数

    def forward_embedding(self, src_tokens, sentence_positions=None, role_positions=None):
        # 该函数用于绝对位置嵌入，其中考虑了标记/轮次/角色嵌入
        # 嵌入标记和位置
        self.embed_tokens = self.embed_tokens.to('cuda')
        src_tokens = src_tokens.to('cuda')
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)

        # 标记绝对位置嵌入
        if self.embed_ap_token is not None:
            x = embed + self.embed_ap_token.forward(src_tokens)

        # 轮次绝对位置嵌入
        if self.embed_ap_sentence is not None and sentence_positions is not None:
            # assert sentence_positions is not None, \
            #     '`sentence_positions` should not be None if `self.embed_positions_sentence` is not None'
            x = x + self.embed_ap_sentence(src_tokens, positions=sentence_positions)

        # 角色绝对位置嵌入
        if self.embed_ap_role is not None and role_positions is not None:
            # assert role_positions is not None, \
            #     '`role_positions` should not be None if `self.embed_positions_role` is not None'
            x = x + self.embed_ap_role(src_tokens, positions=role_positions)

        # layer norm of embeddings
        # x = self.emb_layer_norm(x)
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x, embed

    def forward(
            self, src_tokens, src_lengths=None,
            sentence_positions=None, role_positions=None,
            relative_position_token=None, relative_position_sentence=None,
            masked_tokens=None, **kwargs
    ):
        """
        Args:
            src_tokens: 源语言中的标记，形状为`(batch, src_len)`
            src_lengths: 每个源句子的长度，形状为`(batch)`
            sentence_positions: 句子绝对位置，形状为`(batch, src_len)`
            role_positions: 角色绝对位置，形状为`(batch, src_len)`
            relative_position_token: 标记级相对位置，形状为`(batch, src_len, src_len)`
            relative_position_sentence: 句子级相对位置，形状为`(batch, src_len, src_len)`
            masked_tokens: 掩码位置，形状为`(batch, max_masked_len)`
        """
        x, _ = self.forward_embedding(src_tokens, sentence_positions, role_positions)

        # src_tokens是ID张量
        # words = id_to_word(src_tokens.view(-1), vocab)

        # 如果第0维大于第1维，那么交换两个维度
        # if x.shape[1] == 16 or x.shape[1] == 8:
        #     x = x.transpose(0, 1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # 计算padding掩码
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # if not encoder_padding_mask.any():
        #     encoder_padding_mask = None

        # 编码器层
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layer_drop):
                # 如果不使用相对位置嵌入，直接经过编码器层
                if self.embed_rp_sentence is None and self.embed_rp_token is None:
                    x = layer(x, encoder_padding_mask=encoder_padding_mask)
                # 否则传入相对位置信息relative_position_token和relative_position_sentence
                else:
                    x = layer(
                        x, encoder_padding_mask=encoder_padding_mask,
                        relative_position_token=relative_position_token,
                        relative_position_sentence=relative_position_sentence)

        cls_feature, cls_bow_logits, z, kl, masked_logits, latent_bow_logits = None, None, None, None, None, None

        if self.with_cls_bow_logits:  # 如果包括[CLS]词袋logits
            cls_feature = self.map_cls_to_feature(x)  # 获得CLS标记的隐藏状态的特征
            cls_bow_logits = self.forward_cls_bow_logits(cls_feature)  # 通过[CLS]特征得到词袋logits

        if self.generate_latent_variable:  # 生成隐变量
            if cls_feature is None:
                cls_feature = self.map_cls_to_feature(x)  # 获得CLS标记的隐藏状态的特征
            # 采样隐变量均值和方差
            mu, log_var = self.sample(cls_feature)
            # 连接到潜在空间
            z = self.connect(mu, log_var, self.deterministic)
            # 计算k-l
            if not self.disable_kl_loss:
                kl = self.kl_divergence(mu, log_var)
            if self.with_latent_bow_logits:
                # 将隐变量的隐藏状态映射到一个新的特征空间中，如有需要可以将隐变量与cls特征连接
                latent_feature = self.map_latent_to_feature(z, cls_feature)
                # 通过隐变量特征得到词袋分布
                latent_bow_logits = self.forward_latent_bow_logits(latent_feature)

        if self.with_mask_lm_logits and masked_tokens is not None:
            masked_feature = self.map_masked_to_feature(x, masked_tokens)
            masked_logits = self.forward_masked_lm_logits(masked_feature)

        return {
            'encoder_out': x,                                   # `seq_len, batch_size, embedding_dim`
            'encoder_padding_mask': encoder_padding_mask,       # `batch_size, seq_len`
            'cls_bow_logits': cls_bow_logits,                   # `batch_size, dictionary_dim`
            'z': z,                                             # `batch_size, latent_dim`
            'kl': kl,                                           # `batch_size`
            'latent_bow_logits': latent_bow_logits,             # `batch_size, dictionary_dim`
            'masked_logits': masked_logits,                     # `batch_size, max_masked_len, dictionary_dim`
        }

    @staticmethod
    def re_parameterize(mu, log_var):
        # import torch
        # log_var = torch.zeros(4)
        # 利用重参数化技巧对隐变量进行采样
        std = log_var.mul(.5).exp()  # 计算标准差std，它是log_var乘以0.5再取指数的结果
        eps = torch.zeros_like(std).normal_()
        return mu + torch.mul(eps, std)

    @staticmethod
    def kl_divergence(mu, log_var):
        # 计算需要优化的k-l损失
        # 引用论文:
        # 1. generating sentences from a continuous space
        # 2. improving variational inference with inverse autoregressive flow
        # 3. improving variational encoder-decoders in dialogue generation
        # 4. learning discourse-level diversity for neural dialog models using conditional variational auto-encoders
        kl_loss = .5 * (mu.pow(2.0) + log_var.exp() - log_var - 1.0)
        return kl_loss.sum(dim=1)

    # 将[CLS]特征连接到一个隐藏空间
    def sample(self, feature):
        # 该函数返回潜变量的均值和对数方差
        # 将[CLS]特征转换为均值和对数方差
        # chunk(2, -1)将张量沿着-1维分成2个块。这意味着全连接层的输出被分成两个相等的部分，分别赋值给mu和log_var
        mu, log_var = self.vae_fc2(feature).chunk(2, -1)
        return mu, log_var

    def connect(self, mu, log_var, deterministic):
        # 在推理过程中，如果方差等于零，则采样过程是确定的
        if deterministic:
            log_var.fill_(.0)
        # 重参数化
        z = self.re_parameterize(mu, log_var)
        return z

    # 将[CLS]的隐藏状态映射到一个新的特征空间中
    def map_cls_to_feature(self, x):
        """
        Args:
            x: 输入的隐藏张量，大小为`seq_len, batch, embedding_dim`
        """
        # 映射[CLS]隐藏状态到特征以生成隐变量的均值和对数方差
        hidden = x[0]
        # [CLS]标记的隐藏状态的特征以及后验分布特征
        feature = torch.tanh(self.vae_fc1(hidden))
        return feature

    # 通过[CLS]特征得到词袋logits
    def forward_cls_bow_logits(self, feature):
        # 这个函数返回需要优化的词袋分布，并且feature由编码器[CLS]生成
        assert self.with_cls_bow_logits, '`with_encoder_bow_logits` parameter should be set to true!'
        # 词袋分布
        if self.share_all_embeddings:
            # 如果share_all_embeddings=True，则使用嵌入层的参数来计算词袋分布
            bow_logits = F.linear(feature, weight=self.embed_tokens.weight)
        else:
            assert self.bow_fc_enc is not None, '`self.bow_fc_enc` should not be None!'
            bow_logits = self.bow_fc_enc(feature)
        return bow_logits

    # 将隐变量的隐藏状态映射到一个新的特征空间中，如有需要可以将隐变量与CLS特征连接
    def map_latent_to_feature(self, z, cls_feature):
        """
        Args:
            z: latent variable tensor of shape `batch, latent_dim`
            cls_feature: cls feature of shape `batch, embedding_dim`
        """
        # 如果需要将隐变量与cls特征连接起来
        if self.extend_latent_with_cls:
            hidden = torch.cat([z, cls_feature], dim=1)
        else:
            hidden = z
        feature = torch.tanh(self.vae_fc3(hidden))
        return feature

    # 通过隐变量得到词袋分布
    def forward_latent_bow_logits(self, feature):
        # 这个函数返回需要优化的词袋分布
        # 并且特征是由隐变量生成的，可能还带有编码器特征
        assert self.with_latent_bow_logits, '`with_encoder_latent_logits` parameter should be set to true!'
        # 词袋分布
        if self.share_all_embeddings:
            # 如果share_all_embeddings，则使用嵌入层的参数来计算词袋分布
            bow_logits = F.linear(feature, weight=self.embed_tokens.weight)
        else:
            assert self.bow_fc_latent is not None, '`self.bow_fc_latent` should not be None!'
            bow_logits = self.bow_fc_latent(feature)
        return bow_logits

    # 将掩码标记映射到新的特征空间
    def map_masked_to_feature(self, x, masked_position):
        """
        Args:
            x: 输入隐藏张量，形状为`seq_len, batch, embedding_dim`
            masked_position: 掩码位置张量，形状为`batch, max_masked_len`
        """
        # 将所有掩码标记映射到特征以预测掩码标记
        _, _, embed_dim = x.shape
        # permute函数用于改变张量的维度顺序。在这种情况下，它将x的第一个维度和第二个维度交换，而第三个维度保持不变。
        # 例如，如果 x 的形状为(a, b, c)，那么x.permute(1, 0, 2)的形状将为(b, a, c)。
        # contiguous函数用于返回一个连续的内存块中存储的张量。当对张量进行某些操作(如转置、切片或 permute)时，
        # 它们在内存中的存储方式可能会变得不连续。这可能会影响某些操作的性能，因此有时需要使用 contiguous 函数来返回一个连续存储的副本
        _x = x.permute(1, 0, 2).contiguous()  # [batch, seq_len, embedding_dim]
        # 首先使masked_position形状变为[batch, max_masked_len, embedding_dim]
        # 然后torch.gather函数沿着指定维度根据索引张量中的索引从输入张量中收集值
        # 最后得到的hidden大小也是[batch, max_masked_len, embedding_dim]
        hidden = torch.gather(_x, 1, masked_position.unsqueeze(-1).repeat(1, 1, embed_dim).long())
        # 掩码标记的隐藏状态的特征
        feature = torch.tanh(self.mask_lm_fc1(hidden))
        return feature

    # 通过掩码标记的特征生成掩码标记分布
    def forward_masked_lm_logits(self, feature):
        # 该函数返回需要优化的掩码标记分布
        # 并且该特征是由掩码标记的隐藏状态生成的
        assert self.with_mask_lm_logits, '`with_mask_lm_logits` parameter should be set to true!'
        if self.share_all_embeddings:
            # 如果share_all_embeddings，则使用嵌入层的参数来计算预测的分布
            masked_logits = F.linear(feature, weight=self.embed_tokens.weight)
        else:
            assert self.mask_lm_fc2 is not None, '`self.mask_lm_fc2` should not be None'
            masked_logits = self.mask_lm_fc2(feature)
        return masked_logits

    # 训练
    # 根据new_order重新排列编码器输出中的批次，用于训练到推理的变化
    def reorder_encoder_out(self, encoder_out, new_order):          # -> 推理
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['cls_bow_logits'] is not None:
            encoder_out['cls_bow_logits'] = encoder_out['cls_bow_logits'].index_select(0, new_order)
        if encoder_out['z'] is not None:
            encoder_out['z'] = encoder_out['z'].index_select(0, new_order)
        if encoder_out['kl'] is not None:
            encoder_out['kl'] = encoder_out['kl'].index_select(0, new_order)
        if encoder_out['latent_bow_logits'] is not None:
            encoder_out['latent_bow_logits'] = encoder_out['latent_bow_logits'].index_select(0, new_order)
        if encoder_out['masked_logits'] is not None:
            encoder_out['masked_logits'] = encoder_out['masked_logits'].index_select(0, new_order)
        return encoder_out

    # 获得最大位置
    def max_positions(self):
        if self.embed_ap_token is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_ap_token.max_positions())


# Transformer解码器层(类似于编码器，除了编码器-解码器注意力层)
# 原始论文方法: dropout -> add residual -> layer_norm
# multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
# encoder-decoder-attention sublayer: x = layer_norm(x + dropout(encoder_attn(x))
# position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
#
# tensor2tensor方法: layer_norm -> dropout -> add residual
# multi-head self-attention sublayer: x = x + dropout(self_attn(layer_norm(x))))
# encoder-decoder-attention sublayer: x = x + dropout(encoder_attn(layer_norm(x)))
# position-wise feed-forward sublayer: x = x + dropout(fc2(dropout(relu(fc1(layer_norm(x))))))


# 预测未来n个标记的解码器层
class NgramTransformerDecoderLayer(nn.Module):
    def __init__(
            self, ngram, encoder_embed_dim, decoder_embed_dim, decoder_ffn_embed_dim, decoder_attention_heads,
            dropout, attention_dropout, activation_dropout, activation_fn, with_encoder_decoder_attn
    ):
        super().__init__()

        self.with_encoder_decoder_attn = with_encoder_decoder_attn  # 编码器-解码器注意力
        self.embed_dim = decoder_embed_dim
        self.ngram_self_attn = NgramMultiheadAttention(  # ngram多头注意力
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            ngram=ngram
        )

        self.dropout = dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)  # 激活函数
        self.activation_dropout = activation_dropout

        self.ngram = ngram
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if self.with_encoder_decoder_attn:
            # 编码器-解码器层
            self.encoder_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=decoder_attention_heads,
                kdim=encoder_embed_dim,
                vdim=encoder_embed_dim,
                dropout=attention_dropout,
                self_attention=False,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None

        self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = False

    def forward(
            self, x, encoder_out=None, encoder_mask=None, incremental_state=None, prev_self_attn_state=None,
            prev_attn_state=None, self_attn_mask=None, ngram_mask_matrix=None,
            i_buckets_main_stream=None, i_bucket_relative_stream=None, real_positions=None,
            latent_context=None
    ):
        # multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
        residual = x  # 残差
        # 在增量解码过程中，自注意力层可以使用过去的键和值来计算注意力权重
        if prev_self_attn_state is not None:
            # 检查incremental_state是否为None。如果是，则将其初始化为空字典
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            # 使用self.self_attn._set_input_buffer函数将saved_state设置为self.self_attn的输入缓冲区
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        # 将ngram_self_attn模块统一为MultiheadAttentionRPE模块
        x, attn = self.ngram_self_attn(
            query=x,
            incremental_state=incremental_state,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions,
            latent_context=latent_context,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # 如果训练模式是vae，则不应使用编码器-解码器注意力
        # 解码器应该只依赖于隐变量
        if self.with_encoder_decoder_attn:
            # encoder-decoder-attention sublayer: x = layer_norm(x + dropout(encoder_attn(x)))
            residual = x
            # 在增量解码过程中，自注意力层可以使用过去的键和值来计算注意力权重
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.encoder_attn_layer_norm(x)

        # position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


# 未来n个标记的解码器
class NgramTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        # self.training_mode = args.training_mode
        self.with_encoder_decoder_attn = args.with_encoder_decoder_attn
        self.ngram = args.ngram
        self.num_buckets = args.num_buckets
        self.relative_max_distance = args.relative_max_distance
        self.max_target_positions = args.max_target_positions
        self.max_target_positions_token = args.max_target_positions_token
        self.use_latent_variable = True if args.use_latent_variable else False
        self.dropout = args.dropout

        self.embed_dim = args.decoder_embed_dim
        assert embed_tokens.embedding_dim == self.embed_dim, \
            '`encoder_embed_dim` parameter in global args must equal to the one in self-attention'
        self.decoder_attention_heads = args.decoder_attention_heads
        self.head_dim = self.embed_dim // self.decoder_attention_heads

        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        # self.embed_scale = None
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.decoder_embed_dim)

        # 参数为_input（输入序列），incremental_state（可选，用于增量解码的状态）和positions（可选，预先计算的位置）。
        # 如果positions未提供，则根据输入序列和填充索引计算位置。然后，该方法调用父类的forward方法来获取位置嵌入，
        # 并返回位置嵌入和实际位置。max_positions方法返回模型支持的最大位置数。如果设置了填充索引，
        # 则最大位置数等于嵌入数量减去填充索引再减一；否则，最大位置数等于嵌入数量。
        self.embed_ap_token = LearnedPositionalEmbeddingNew(
            self.max_target_positions_token + 2 + self.padding_idx, self.embed_dim, self.padding_idx,
        )

        self.ngram_input_embed = Embedding(self.ngram, self.embed_dim, None)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NgramTransformerDecoderLayer(
                args.ngram,
                args.encoder_embed_dim,
                self.embed_dim,
                args.decoder_ffn_embed_dim,
                args.decoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
                self.with_encoder_decoder_attn
            )
            for _ in range(args.decoder_layers)
        ])

        self.emb_layer_norm = LayerNorm(self.embed_dim)

        # if not self.with_encoder_decoder_attn:
        #     self.vae_transform = nn.Linear(
        #         in_features=args.latent_size, out_features=2 * self.embed_dim)
        # else:
        #     self.vae_transform = None

        # 将VAE中的隐变量合并到解码器，乘以2，因为生成键和值
        if self.use_latent_variable:
            self.vae_transform = nn.Linear(
                in_features=args.latent_size, out_features=2 * self.embed_dim)
        else:
            self.vae_transform = None
        self.apply(init_bert_params)  # 用于在使用fairseq库构建BERT模型时初始化模型的参数

    def forward(
            self, prev_output_tokens, encoder_out=None,
            incremental_state=None, vae_hidden=None, **kwargs
    ):
        x_list, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **kwargs)
        x_predicted = x_list[1:]
        # x_predicted = x_list
        x_predicted = [self.output_layer(x) for x in x_predicted]
        # x_predicted = self.output_layer(x_predicted)

        if incremental_state is not None:
            x_predicted = x_predicted[0]
            for k in extra:
                if extra[k] is not None:
                    extra[k] = extra[k][0]
        return x_predicted, extra

    # 相对位置桶
    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    # 计算预训练相对位置
    def cal_pretrain_relative_positions(self, real_positions):
        main_stream_relative_positions = real_positions.unsqueeze(1)
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main
        real_positions_shift_predicting_stream = real_positions - 1
        predicting_stream_relative_positions = torch.cat((
            real_positions_shift_predicting_stream, real_positions), dim=-1).unsqueeze(1)
        predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(
            1, real_positions.size(-1), 1)
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(
            predicting_stream_relative_positions, bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream

    # 计算精调相对位置
    def cal_finetune_relative_positions(self, real_positions):
        n_tokens = real_positions.size(-1)
        batch_size = real_positions.size(0)

        if not hasattr(self, '_finetune_i_bucket_main_stream') or \
                self._finetune_i_bucket_main_stream is None or \
                self._finetune_i_bucket_main_stream.device != real_positions.device:
            fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            self._finetune_i_bucket_main_stream = finetune_i_bucket_main_stream.to(real_positions.device)
            self._finetune_i_bucket_predicting_stream = finetune_i_bucket_predicting_stream.to(real_positions.device)

        finetune_i_bucket_main_stream = self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens].repeat(
            batch_size, 1, 1)
        finetune_i_bucket_predicting_stream = torch.cat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2).repeat(batch_size, 1, 1)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream

    # Memory Scheme：隐变量z被映射到一个额外的内存向量，记为hMem，它是解码器要关注的额外键值对
    def transform_latent_context(self, latent_context):
        # 将隐变量转化为潜在键和值
        # latent_context -> [batch_size, latent_dim]
        latent_context = self.vae_transform(latent_context)
        # latent_context -> [batch_size, 2 * embedding_dim]]
        latent_context = latent_context.view(-1, 1, 2 * self.head_dim)
        # latent_context -> [batch_size * num_heads, 1, 2 * head_dim
        # 沿着最后一个维度将latent_context分割为大小为self.head_dim的块
        latent_context = torch.split(latent_context, self.head_dim, dim=-1)
        # latent_context -> [batch_size * num_heads, 1, self.head_dim]
        return latent_context

    # 提取特征
    def extract_features(
            self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):

        main_stream_pos_embed, real_positions = self.embed_ap_token(  # 获取主流绝对位置嵌入和真实位置
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_ap_token is not None else None

        # 如果使用增量解码，则不需要计算桶位置
        if incremental_state is not None:
            i_buckets_main_stream, i_bucket_relative_stream = None, None
        # 否则要计算精调相对位置
        else:
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_finetune_relative_positions(real_positions)

        # 计算预测流位置嵌入
        predicting_stream_pos_embed = self.embed_ap_token._forward(real_positions + 1)

        if incremental_state is not None:
            # 请理解，在增量解码中，只需要最后一个输出进行解码
            prev_output_tokens = prev_output_tokens[:, -1:]
            # 主流的绝对位置嵌入也只取最后一个
            if main_stream_pos_embed is not None:
                main_stream_pos_embed = main_stream_pos_embed[:, -1:]

        # 计算x的标记嵌入
        x = self.embed_tokens(prev_output_tokens)
        if self.embed_scale is not None:
            x *= self.embed_scale

        # x标记嵌入加上主流位置嵌入
        if main_stream_pos_embed is not None:
            x += main_stream_pos_embed

        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]  # 将内部状态设为x
        # 位置应该用于预测ngrams
        if main_stream_pos_embed is None:
            print('positions should be used to predict ngrams')
            raise Exception()

        # 嵌入缩放
        if self.embed_scale is not None:
            ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
        else:
            ngram_input_embed = self.ngram_input_embed.weight
        # ngram embedding

        # 如果使用增量解码
        if incremental_state is not None:
            B = x.size(1)  # B为x的第二个维度的大小
            # 将n-gram的输入嵌入与预测流位置嵌入相加，然后转置并重复B次
            ngram_masks = [
                (ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
                for ngram in range(self.ngram)]
        # 否则直接相加并转置
        else:
            ngram_masks = [(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1) for
                           ngram in range(self.ngram)]

        # 如果使用增量解码，就直接使用缓存的掩码
        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
        ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None

        x = torch.cat([x] + ngram_masks, 0)  # 将x与ngram掩码连接

        # 层归一化
        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Memory Scheme
        # 将隐变量转换为ProphetNet中使用的主流的第一个键和值
        # 向解码器注入隐变量，即解码器中的自注意力
        # new_attn = MutliHead(q, [k, k0], [v, v0])
        #
        # key0, value0   key1, value1   key2, value2   key3, value3
        #                   token1         token2        token3
        #
        # 预测流可以通过主流的隐藏状态隐含地使用隐变量
        # 潜在键和值的参数在所有解码器层中共享

        if not self.use_latent_variable:
            latent_context = None
        # 将隐变量z映射到一个额外的内存向量
        else:
            # latent_context -> [batch_size * num_heads, 1, self.head_dim]
            latent_context = self.transform_latent_context(encoder_out['z'])

        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                ngram_mask_matrix=ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions,
                # 具有形状[batch_size*num_heads，1，head_dim]的2元素张量元组
                latent_context=latent_context,
            )
            inner_states.append(x)

        # chunk函数的第一个参数1+self.ngram指定了要创建的块的数量，第二个参数1指定了沿着哪个维度进行分割
        # print("x:", x)
        # print("attn:", attn)
        x_list = x.transpose(0, 1).chunk(1 + self.ngram, 1)
        if attn is not None:
            # chunk将张量沿着1维分成ngram+1块
            attn_list = attn.transpose(0, 1).chunk(1 + self.ngram, 1)
        else:
            attn_list = None

        return x_list, {'attn': attn_list}

    # 获得归一化概率
    def get_normalized_probs(self, net_output, log_probs, sample):
        # 如果有'adaptive_softmax'属性且不为空
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:  # 如果样本中有target
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            # 使用adaptive_softmax的get_log_prob方法获取对数概率
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        # 如果该类不具有名为adaptive_softmax的属性或其值为None，则从net_output[0]中获取logits。
        # 然后，根据log_probs参数的值，使用utils模块中的log_softmax或softmax函数对logits进行归一化处理
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    # F.linear函数执行矩阵乘法，将输入矩阵features与权重矩阵self.embed_tokens.weight相乘，然后返回结果。
    # 这意味着该函数将输入特征映射到一个新的空间，通常用于预测或分类任务
    def output_layer(self, features, **kwargs):
        return F.linear(features, self.embed_tokens.weight)

    def max_positions(self):
        # 如果没有使用绝对位置嵌入，直接返回目标序列的最大位置
        if self.embed_ap_token is None:
            return self.max_target_positions_token
        # 否则返回目标序列的最大位置和绝对位置最大值中的较小值
        return min(self.max_target_positions, self.embed_ap_token.max_positions())

    # 缓存未来序列掩码
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)  # 存储张量第1维大小
        # 检查对象是否具有名为_future_mask的属性，以及该属性是否为 None，是否与tensor在同一设备上，以及其第0维的大小是否小于dim
        # 如果满足这些条件中的任何一个，则使用torch.triu和utils.fill_with_neg_inf创建未来序列掩码
        if not hasattr(self, '_future_mask') or \
                self._future_mask is None or \
                self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        # 返回_future_mask张量的前dim行和前dim列
        return self._future_mask[:dim, :dim]

    # # 缓存未来ngram序列掩码
    def buffered_future_mask_ngram(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_ngram_future_mask') or \
                self._ngram_future_mask is None or \
                self._ngram_future_mask.device != tensor.device:
            self._ngram_future_mask = ngram_attention_bias(
                self.max_target_positions_token, self.ngram).type(tensor.dtype).to(tensor.device)
        ngram_future_mask = torch.cat([
            self._ngram_future_mask[:, :dim, :dim],
            self._ngram_future_mask[:, :dim, self.max_target_positions_token: self.max_target_positions_token + dim]], 2)
        return ngram_future_mask


def base_architecture(args):
    # fairseq中的默认参数不能在该函数中修改两次，初始化时默认会存在。
    # 如果默认参数需要在命令行中修改，请直接在该函数中将其分配，如下所示：

    # 编码器结构
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_layer_drop = getattr(args, 'encoder_layer_drop', 0.0)

    # 编码器位置嵌入
    args.with_encoder_ape_token = getattr(args, 'with_encoder_ape_token', True)

    # default is False
    args.with_encoder_ape_sentence = getattr(args, 'with_encoder_ape_sentence', True)
    args.with_encoder_ape_role = getattr(args, 'with_encoder_ape_role', True)

    # 编码器绝对位置嵌入
    # args.max_source_positions_token = getattr(args, 'max_source_positions_token', args.max_source_positions)
    args.max_source_positions_token = getattr(args, 'max_source_positions_token', 512)
    # 假设对话的总轮次不超过32次
    args.max_source_positions_sentence = getattr(args, 'max_source_positions_sentence', 32)
    # 1表示回答，2表示其他角色，3表示可能的背景知识
    args.max_source_positions_role = getattr(args, 'max_source_positions_role', 4)

    # 编码器相对位置嵌入
    args.num_buckets_source_token = getattr(args, 'num_buckets_source_token', 32)
    args.max_distance_source_token = getattr(args, 'max_distance_source_token', 128)
    args.bidirectional_source_token = getattr(args, 'bidirectional_source_token', True)

    args.num_buckets_source_sentence = getattr(args, 'num_buckets_source_sentence', 4)
    args.max_distance_source_sentence = getattr(args, 'max_distance_source_sentence', 16)
    args.bidirectional_source_sentence = getattr(args, 'bidirectional_source_sentence', True)

    # default is False
    args.with_encoder_rpe_token = getattr(args, 'with_encoder_rpe_token', False)
    args.with_encoder_rpe_sentence = getattr(args, 'with_encoder_rpe_sentence', False)

    # 解码器结构
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)

    # 解码器位置嵌入
    args.max_target_positions_token = getattr(args, 'max_target_positions_token', 512)

    # 解码器相对位置嵌入
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    # common
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
    args.deterministic = getattr(args, 'deterministic', False)


# ----------------------------------------------------------------------------------------------------------------------
# 发布的预训练模型

# masked language loss + response generation loss
# DialogVED-Seq2Seq模型没有隐变量，它是一个与DialogVED一样具有相同训练设置的纯Seq2Seq模型
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq_pretrain')
def transformer_seq2seq_pretrain(args):
    # 该参数是基于seq2seq的模型和基于vae的模型之间的关键区别
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', False)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', False)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)


# masked language loss + response generation loss + bag-of-word loss + K-L loss
# 隐变量大小为32的DialogVED-VAE-Standard模型
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_standard_pretrain')
def transformer_vae_standard_pretrain(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', True)
    args.latent_size = getattr(args, 'latent_size', 32)


# masked language loss + response generation loss + bag-of-word loss + K-L loss
# 隐变量大小为64的DialogVED-VAE-Large模型
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_large_pretrain')
def transformer_vae_large_pretrain(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', False)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', True)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', True)
    args.latent_size = getattr(args, 'latent_size', 64)


# ----------------------------------------------------------------------------------------------------------------------
# 发布的经过微调的模型

@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq')
def transformer_seq2seq(args):
    # this parameter is the key difference between seq2seq-based and vae-based models
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', False)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', False)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_standard')
def transformer_vae_standard(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)

    # 标准VAE transformer的隐变量大小等于32
    args.latent_size = getattr(args, 'latent_size', 32)


# 使用的模型架构
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_large')
def transformer_vae_large(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', True)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)

    # 大型VAE transformer的隐变量大小等于64
    args.latent_size = getattr(args, 'latent_size', 64)
