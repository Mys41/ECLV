import torch
import torch.nn as nn
from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 话语编码器
class UtterEncoder(nn.Module):
    def __init__(self, config, model_size, utter_dim, conv_encoder='none', rnn_dropout=None):
        super(UtterEncoder, self).__init__()
        self.encoder = RobertaModel(config)
        self.mapping = nn.Linear(config.hidden_size, utter_dim)
        if conv_encoder == 'none':
            self.register_buffer('conv_encoder', None)
        else:
            self.conv_encoder = nn.GRU(input_size=utter_dim,
                                       hidden_size=utter_dim,
                                       bidirectional=True,
                                       num_layers=1,
                                       dropout=rnn_dropout,
                                       batch_first=True)

    def forward(self, conv_utterance, attention_mask, conv_len, type='utt'):
        # conv_utterance: [[conv_len1, max_len1], [conv_len2, max_len2], ..., [conv_lenB, max_lenB]]
        processed_output = []
        if type != 'csk':
            for cutt, amsk in zip(conv_utterance, attention_mask):
                # cutt = conv_utterance.get('1').to('cuda')
                # amsk = attention_mask.get('1').to('cuda')
                # cutt_emb = self.embedding(cutt)

                # 将对话话语和对应的掩码输入Roberta模型进行编码，并输出最后一层的隐藏状态作为话语表示
                output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state
                # output_data = self.encoder(cutt_emb, amsk)
                # [conv_len, token_dim] -> [conv_len, utter_dim]
                # 使用torch.max函数沿着第一个维度对隐藏状态取最大值，得到池化输出
                # torch.max函数返回一个元组，其中第一个元素是最大值，第二个元素是最大值所在的索引。
                # 在这行代码中，我们只关心最大值，因此使用[0]来获取元组中的第一个元素
                pooler_output = torch.max(output_data, dim=1)[0]
                # pooler_output = self.pooler(output_data)
                # pooler_output = torch.max(output_data, dim=1)[0]
                # 使用线性层将池化输出pooler_output映射到一个新的空间
                mapped_output = self.mapping(pooler_output)
                processed_output.append(mapped_output)
            # [batch_size, conv_size, utter_dim]
            conv_output = pad_sequence(processed_output, batch_first=True)
        else:
            for cutt, amsk in zip(conv_utterance, attention_mask):
                output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state
                # [conv_len, token_dim] -> [conv_len, utter_dim]
                pooler_output = torch.max(output_data, dim=1)[0]
                processed_output.append(pooler_output)
            # [batch_size, conv_size, utter_dim]
            conv_output = pad_sequence(processed_output, batch_first=True)

        if self.conv_encoder is not None:
            pad_conv = pack_padded_sequence(conv_output, conv_len, batch_first=True, enforce_sorted=False)
            pad_output = self.conv_encoder(pad_conv)[0]
            conv_output = pad_packed_sequence(pad_output, batch_first=True)[0]

        return conv_output
