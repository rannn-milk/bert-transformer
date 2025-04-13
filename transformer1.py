import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism.
   input_dim: batch_size*3*session_length
   out_dim: batch_size*3*session_length
    """

    def __init__(
            self, n_type_correctness, n_type_problem, n_type_skill, embedding_size, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_type_qno, n_type_kno, dropout=0.1, n_position=200):

        super().__init__()
        self.problem_emb = nn.Embedding(n_type_problem, embedding_size)
        self.skill_emb = nn.Embedding(n_type_skill, embedding_size)
        self.correctness_emb = nn.Embedding(n_type_correctness, embedding_size)
        self.qno_emb = nn.Embedding(int(n_type_qno), int(embedding_size))
        self.position_enc = PositionalEncoding(embedding_size, n_position=n_position, encoder_type='Action')
        self.dropout = nn.Dropout(p=dropout)
        # layer_stack包含多个编码层的列表
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, encoder_type='Action')
            for _ in range(n_layers)])
        # 编码器后就接layernorm层归一化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, input_array,emb, pad_mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        # element wise addition on correctness embedding and problemID embedding
        problemId_embedding = self.problem_emb(input_array[:, :, 0])  # 第三维的第一个切片
        skill_embedding = self.skill_emb(input_array[:, :, 1])
        #correct_embedding = self.correctness_emb(input_array[:, :, 2])
        qestion_embedding= emb
        qno_embedding = self.qno_emb(input_array[:, :, 3])

        enc_output = qestion_embedding + problemId_embedding + skill_embedding + qno_embedding

        position_encoding = self.position_enc(enc_output)

        enc_output = self.dropout(position_encoding)
        enc_output = self.layer_norm(enc_output)

        # 遍历enc_layer的每一层
        # enc_output 是当前层的输出，它将作为下一个层的输入
        # slf_attn_mask=pad_mask 用于处理填充值，确保填充值在计算自注意力时不会影响结果
        # enc_slf_attn 是当前层计算的自注意力权重，可以用来分析模型在处理输入时关注的部分
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=pad_mask)
            # 如果enc_slf_attn为真，则将当前层的自注意力权重 enc_slf_attn 添加到 enc_slf_attn_list 中
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, embedding_size, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_type_problem, n_type_skill, n_type_qno, n_type_kno, n_position=200, dropout=0.1):

        super().__init__()
        self.problem_emb = nn.Embedding(n_type_problem, embedding_size)
        self.skill_emb = nn.Embedding(n_type_skill, embedding_size)
        self.qno_emb = nn.Embedding(int(n_type_qno), int(embedding_size))
        self.position_enc = PositionalEncoding(embedding_size, n_position=n_position, encoder_type='Dec')
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, encoder_type='Dec')
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_problem_seq, trg_skill_seq, trg_mask, pad_mask,
                enc_output, target_qno, target_kno, return_attns=False):

        dec_slf_attn_list = []

        # -- target seq embedding
        problemId_embedding = self.problem_emb(trg_problem_seq)
        skill_embedding = self.skill_emb(trg_skill_seq)
        qno_embedding = self.qno_emb(target_qno)

        # -- Forward
        dec_output = problemId_embedding + skill_embedding + qno_embedding

        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_input=dec_output, enc_output=enc_output, slf_attn_mask=trg_mask,
                                                 dec_enc_attn_mask=pad_mask)

            dec_slf_attn_list += [dec_slf_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list
        return dec_output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, encoder_type='Action'):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, encoder_type=encoder_type)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask
                                                 )
        # slf-attention后就接着ffn前馈神经网络
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200, encoder_type='Action'):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.encoder_type = encoder_type

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        # 计算给定位置的编码向量
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        if self.encoder_type == 'Action':
            return x + self.pos_table[:, None, :x.size(2)].clone().detach()
        else:
            return x + self.pos_table[:, :x.size(1)].clone().detach()


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, encoder_type='Action'):
        super(DecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, encoder_type=encoder_type)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)

        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout, encoder_type):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.encoder_type = encoder_type

        # 线性层将输入的特征维度（d_model）映射到多个头的维度
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, Time_affect=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        if (self.encoder_type != 'Action'):
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        else:
            sz_b, ses_L, len_q, len_k, len_v = q.size(0), q.size(1), q.size(2), k.size(2), v.size(2)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if (self.encoder_type != 'Action'):
            # 通过view方法重新组织为多个头的形式
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        else:
            q = self.w_qs(q).view(sz_b, ses_L, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, ses_L, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, ses_L, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        if self.encoder_type != 'Action':
            # 通过转置操作，将输入转换为适合进行注意力计算的形状，这种操作确保了注意力计算时能够在不同的头之间进行操作
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None and self.encoder_type != 'Action':
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        elif mask is not None and self.encoder_type == 'Action':
            mask = mask.unsqueeze(2)  # For head axis broadcasting.

        if Time_affect == False:
            # 注意力计算
            q, attn = self.attention(q, k, v, mask=mask, encoder_type=self.encoder_type)
        #else:
           # q, attn = DIY_attention(q, k, v, d_k=self.d_k, mask=mask, dropout=self.dropout)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        if (self.encoder_type != 'Action'):
            # 头的输出被转置回原来的形状，并合并到一起
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        else:
            q = q.transpose(2, 3).contiguous().view(sz_b, ses_L, len_q, -1)
        q = self.dropout(self.fc(q))  #全连接层 self.fc
        # 残差连接，将输入 q 直接加到输出上，可以保留原始信息，避免梯度消失
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise，输入层到隐藏层
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise，隐藏层返回到输入层
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)  # 层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    # 计算注意力分数
    def forward(self, q, k, v, mask=None, encoder_type='Action'):
        if encoder_type != 'Action':
            # 使用 torch.matmul 计算查询与键的点积
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # bs*head*seq_l*seq_l

        else:
            attn = torch.matmul(q / self.temperature, k.transpose(3, 4))  # bs*head*seq_l*seq_l

        #if mask is not None:
            # 将掩码值为0的位置的注意力分数设置为非常小的值
         #   attn = attn.masked_fill(mask == 0, -1e32)

        # 计算注意力权重
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
