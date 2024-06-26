import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.encoder import UtterEncoder
from transformers import RobertaConfig


class CausePredictor(nn.Module):
    def __init__(self, input_dim, mlp_dim, mlp_dropout=0.1):
        super(CausePredictor, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        # 义了一个多层感知器，它包含两个线性层，每个线性层之间都有一个ReLU激活函数和一个Dropout层。
        # 然后，定义了一个预测器权重，它是一个线性层，用来将多层感知器的输出转换为一个标量值。这个标量值可以用来进行预测。
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_dim, False).to(torch.float64), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False).to(torch.float64), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, 1, False).to(torch.float64)

    def forward(self, x, conv_len, mask):
        # 使用torch.sigmoid函数将预测分数转换为 0 到 1 之间的值
        predict_score = self.predictor_weight(self.mlp(x)).squeeze(-1)
        predict_score = predict_score
        predict_score = torch.sigmoid(predict_score) * mask.to('cuda')

        return predict_score


class CSK_Measure(nn.Module):
    def __init__(self, opt, in_features, hidden_features, attention_probs_dropout_prob=0.1):
        super(CSK_Measure, self).__init__()
        self.d = hidden_features
        self.opt = opt
        self.weight = nn.Linear(in_features, hidden_features)
        self.query = nn.Linear(in_features, hidden_features).to(torch.float64)
        self.key = nn.Linear(in_features, hidden_features).to(torch.float64)
        self.value = nn.Linear(in_features, hidden_features).to(torch.float64)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, target_vector, hidden_vectors, target_emotion, intra_csk, inter_csk, intra_mask, inter_mask,
                type='emotion'):
        batch_size, seq_len, x_dim = hidden_vectors.size()
        target_vector = target_vector.to(torch.float64)
        target_emotion = target_emotion.to(torch.float64)
        if type == 'emotion' and len(target_emotion) > 0:
            target_emotion = target_emotion.unsqueeze(-1).repeat(1, 1, 300)
            query_layer_intra = self.query(target_vector + target_emotion)  # (B, 1, D)
            query_layer_inter = self.query(target_vector + target_emotion)
        else:
            query_layer_intra = self.query(target_vector)
            query_layer_inter = self.query(target_vector)

        # 通过线性层时，可能会由于权重矩阵而改变数据类型
        key_layer_intra = self.key(hidden_vectors) + self.weight(intra_csk)
        value_layer_intra = self.value(hidden_vectors) + self.weight(intra_csk)

        key_layer_inter = self.key(hidden_vectors) + self.weight(inter_csk)
        value_layer_inter = self.value(hidden_vectors) + self.weight(inter_csk)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_intra = torch.matmul(query_layer_intra, key_layer_intra.transpose(-1, -2))
        attention_scores_intra = attention_scores_intra / math.sqrt(self.d)
        attention_scores_intra = attention_scores_intra * intra_mask

        attention_scores_inter = torch.matmul(query_layer_inter, key_layer_inter.transpose(-1, -2))
        attention_scores_inter = attention_scores_inter / math.sqrt(self.d)
        attention_scores_inter = attention_scores_inter * inter_mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        mask = intra_mask + inter_mask
        attention_scores_both = attention_scores_intra + attention_scores_inter

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores_both.masked_fill(mask == 0, -1e9))  # (B, 1, L)
        attention_probs = self.dropout(attention_probs)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # query_layer_intra[2, 3, 300]
        target_enhanced = (attention_probs * intra_mask).transpose(1, 2) * query_layer_intra.expand(batch_size,
                          seq_len, x_dim) + (attention_probs * inter_mask).transpose(1, 2) * \
                          query_layer_inter.expand(batch_size, seq_len, x_dim)

        csk_enhanced = (attention_probs * intra_mask).transpose(1, 2) * value_layer_intra + (
                attention_probs * inter_mask).transpose(1, 2) * value_layer_inter
        final_features = target_enhanced + csk_enhanced

        return final_features, attention_probs


# 交互
class Interaction(nn.Module):
    def __init__(self, opt, in_features, out_features, dropout, alpha, concat=True):
        super(Interaction, self).__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        if opt.use_emo_csk:
            self.emo_csk_interaction = CSK_Measure(opt, in_features, out_features, attention_probs_dropout_prob=0.1)
        if opt.use_act_csk:
            self.int_csk_interaction = CSK_Measure(opt, in_features, out_features, attention_probs_dropout_prob=0.1)
        if opt.use_event_csk:
            self.csk_weight = nn.Linear(in_features, out_features)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, conv_len, emo_vector, event_csk_before, event_csk_after, emo_intra_csk, emo_inter_csk,
                int_intra_csk, int_inter_csk, intra_mask, inter_mask):
        batch_size = inp.shape[0]

        # -------------------------------------CSK-Enhanced Graph Attention-------------------------------------
        h = torch.matmul(inp, self.W)
        N = h.size()[1]
        if self.opt.use_event_csk:
            event_csk_after = self.csk_weight(event_csk_after)
            event_csk_before = self.csk_weight(event_csk_before)

            a_input_intra = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features),
                                       (h + event_csk_after + event_csk_before).repeat(1, N, 1)], dim=-1).view(-1, N, N,
                                                                                                               2 * self.out_features)
            a_input_inter = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features),
                                       (h + event_csk_after + event_csk_before).repeat(1, N, 1)], dim=-1).view(-1, N, N,
                                                                                                               2 * self.out_features)
        else:
            a_input_intra = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)],
                                      dim=-1).view(-1, N, N, 2 * self.out_features)
            a_input_inter = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)],
                                      dim=-1).view(-1, N, N, 2 * self.out_features)
        # [B, N, N, 2*out_features]

        e_intra = self.leakyrelu(torch.matmul(a_input_intra, self.a).squeeze(3))
        e_intra = e_intra * intra_mask

        e_inter = self.leakyrelu(torch.matmul(a_input_inter, self.a).squeeze(3))
        e_inter = e_inter * inter_mask

        adj = intra_mask + inter_mask
        e = e_intra + e_inter

        attention = F.softmax(e.masked_fill(adj == 0, -1e9), dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = h.to(torch.float64)
        h_prime = torch.matmul(attention, h)
        # ---------------------------------------------------------------------------------------------------------

        # -------------------------------------Emotional & Actional Interaction------------------------------------
        target_emotion = []
        target_vector = []
        target_intra_mask = []
        target_inter_mask = []
        for i in range(batch_size):
            if self.opt.add_emotion:
                target_emotion.append(emo_vector[i][conv_len.item() - 1].unsqueeze(0))
            # target_vector.append(inp[i][conv_len[0] - 1].unsqueeze(0))  # inp
            target_vector.append(inp[i][conv_len.item() - 1].unsqueeze(0))
            target_intra_mask.append(intra_mask[i][conv_len.item() - 1].unsqueeze(0))
            target_inter_mask.append(inter_mask[i][conv_len.item() - 1].unsqueeze(0))

        if self.opt.add_emotion:
            target_emotion = torch.cat(target_emotion, dim=0).unsqueeze(1).cuda()
        target_vector = torch.cat(target_vector, dim=0).unsqueeze(1).cuda()
        target_intra_mask = torch.cat(target_intra_mask, dim=0).unsqueeze(1).cuda()
        target_inter_mask = torch.cat(target_inter_mask, dim=0).unsqueeze(1).cuda()

        if self.opt.use_emo_csk:
            emo_connection, emo_attention_probs = self.emo_csk_interaction(target_vector, h_prime, target_emotion,
                                                                           emo_intra_csk, emo_inter_csk,
                                                                           target_intra_mask, target_inter_mask,
                                                                           type='emotion')
        if self.opt.use_act_csk:
            int_connection, act_attention_probs = self.int_csk_interaction(target_vector, h_prime, target_emotion,
                                                                           int_intra_csk, int_inter_csk,
                                                                           target_intra_mask, target_inter_mask,
                                                                           type='action')
        # ----------------------------------------------------------------------------------------------------------

        out = h_prime + inp
        if self.opt.use_emo_csk:
            out = out + emo_connection
        if self.opt.use_act_csk:
            out = out + int_connection

        if self.concat:
            return F.relu(out), emo_attention_probs, act_attention_probs
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 知识桥接的因果交互块
class KBCI(nn.Module):
    def __init__(self, opt, n_feat, n_hid, csk_features, dropout, n_heads, alpha=0.01):
        """Knowledge Bridged Causal Interaction Block
        """
        super(KBCI, self).__init__()
        self.dropout = dropout
        self.opt = opt
        self.n_heads = n_heads
        self.attentions = [Interaction(opt, n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, conv_len, emo_vector, event_csk_before, event_csk_after, emo_intra_csk, emo_inter_csk,
                int_intra_csk, int_inter_csk, intra_mask, inter_mask):
        x = F.dropout(x, self.dropout, training=self.training)
        x_total, emo_attentions, act_attentions = [], [], []
        for att in self.attentions:
            x_i, emo_attention, act_attention = att(x, conv_len, emo_vector, event_csk_before, event_csk_after,
                                                    emo_intra_csk, emo_inter_csk, int_intra_csk, int_inter_csk,
                                                    intra_mask, inter_mask)
            x_total.append(x_i)
            emo_attentions.append(emo_attention)
            act_attentions.append(act_attention)
        x_total = torch.cat(x_total, dim=2)

        x_total = F.dropout(x_total, self.dropout, training=self.training)

        return x_total, emo_attentions, act_attentions


class KBCIN(nn.Module):
    def __init__(self, opt, emotion_num):

        super(KBCIN, self).__init__()
        self.opt = opt
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_hidden_layers = 10
        config.num_attention_heads = 8
        config.hidden_size = 768
        # 使用roberta模型作为话语编码器
        self.utter_encoder = UtterEncoder(config, opt.model_size, opt.hidden_dim, opt.conv_encoder, opt.rnn_dropout)
        if opt.use_pos:
            # 位置嵌入层，最多可以编码32个不同的位置
            self.position_embeddings = nn.Embedding(32, opt.hidden_dim, padding_idx=31)
            # 使用uniform_函数对嵌入层的权重进行初始化，将其初始化为 -0.1 到 0.1 之间的均匀分布
            self.position_embeddings.weight.data.uniform_(-0.1, 0.1)

        # 如果要使用任意的常识知识，就加上常识知识线性层
        if opt.use_emo_csk or opt.use_act_csk or opt.use_event_csk:
            self.csk_lin = nn.Linear(opt.csk_dim, opt.hidden_dim)

        # 知识桥接的因果交互块
        self.interaction = KBCI(opt, opt.hidden_dim, opt.hidden_dim, opt.hidden_dim, dropout=0.1, n_heads=2)

        # 使用mlp预测情感原因语句
        self.classifier = CausePredictor(3 * opt.hidden_dim, 3 * opt.hidden_dim, mlp_dropout=opt.mlp_dropout)
        # 添加情感嵌入
        if opt.add_emotion:
            self.emotion_embeddings = nn.Embedding(emotion_num, opt.emotion_dim, padding_idx=0)

    def forward(self, input_ids, attention_mask, emotion_label, relative_position, intra_mask, inter_mask, bf, af, xW,
                xR, oW, oR, speaker_mask, umask):
        # batch, conv_len, utter_len = input_ids.size()
        text_len = torch.sum(umask != 0, dim=-1).cpu()
        utter_emb = self.utter_encoder(input_ids, attention_mask, text_len)  # conv_len

        # 加上位置嵌入
        if self.opt.use_pos:
            position_emb = self.position_embeddings(relative_position)
            utter_emb = utter_emb + position_emb

        # 加上情感嵌入
        if self.opt.add_emotion:
            emo_emb = self.emotion_embeddings(emotion_label)
            utter_emb = utter_emb + emo_emb
        else:
            emo_emb = None

        inter_features = utter_emb

        # 使用情感常识知识
        if self.opt.use_emo_csk:
            emo_csk_intra = F.relu(self.csk_lin(xR))
            emo_csk_inter = F.relu(self.csk_lin(oR))
        else:
            emo_csk_intra = None
            emo_csk_inter = None

        # 使用行为常识知识
        if self.opt.use_act_csk:
            act_csk_intra = F.relu(self.csk_lin(xW))  # torch.cat([x1, x4], dim=-1)
            act_csk_inter = F.relu(self.csk_lin(oW))
        else:
            act_csk_intra = None
            act_csk_inter = None

        # 使用事件常识知识
        if self.opt.use_event_csk:
            event_csk_before = F.relu(self.csk_lin(bf))
            event_csk_after = F.relu(self.csk_lin(af))
        else:
            event_csk_before, event_csk_after = None, None
        intra_mask = intra_mask.unsqueeze(0)
        # 将常识知识和话语表示输入到因果交互块，输出最终的话语表示以及情感注意力和行为注意力
        final_features, e_att, a_att = self.interaction(inter_features, text_len, emo_emb, event_csk_before,
                                                        event_csk_after, emo_csk_intra, emo_csk_inter, act_csk_intra,
                                                        act_csk_inter, intra_mask, inter_mask)
        final_features = torch.cat([inter_features, final_features], dim=-1)

        # 将话语表示和长度和掩码输入情感原因预测器(mlp)中，获得logits
        logits = self.classifier(final_features, text_len, umask)

        return logits, e_att, a_att
