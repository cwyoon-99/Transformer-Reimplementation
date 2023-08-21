import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super(PositionalEncoding, self).__init__()

        even_index = torch.arange(0, hidden_size, 2) # [0,2,4,6, ...]
        _partial = 10000 ** (even_index / hidden_size)

        self.positional_encoding = torch.zeros(max_len, hidden_size) # seq_len x hidden_size
        self.requires_grad = False # positional encoding is not training parameters

        for pos in range(max_len):
            self.positional_encoding[pos,0::2] = torch.sin(pos / _partial)
            self.positional_encoding[pos,1::2] = torch.cos(pos / _partial)

    def forward(self, x):
        # x: batch x seq_len 
        seq_len = x.size(1)
        return self.positional_encoding[:seq_len, :].unsqueeze(0) # 1 x seq_len x hidden_size

class InputEmbedding(nn.Module):
    def __init__(self, pad_idx, vocab_size, max_len, hidden_size):
        super(InputEmbedding, self).__init__()
        self.src_embedding = nn.Embedding(num_embedding = vocab_size,
                                     embedding_dim = hidden_size,
                                     padding_idx = pad_idx)
        
        self.positional_encoding = PositionalEncoding(max_len, hidden_size)

    def forward(self, x):
        # x: batch  x seq_len
        input_embedding = self.src_embedding(x) # batch x seq_len x hidden_size
        return input_embedding + self.positional_encoding(x) # add positional encoding to input embedding
    
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#         self.softmax = nn.Softmax(dim = -1) # specify the dimension to compute softmax

#     def forward(self, Q, K, V):
#         # batch x seq_len x d_head
#         d_k = K.size(2)
#         K_transpose = K.permute(0,2,1)

#         attention = torch.bmm(Q,K_transpose) / math.sqrt(d_k) # batch x seq_len x seq_len
#         softmax = self.softmax(attention) # batch x seq_len x seq_len

#         return torch.bmm(softmax, V) # batch x seq_len x d_head


# class SingleHeadAttention(nn.Module):
#     def __init__(self, hidden_size, d_head):
#         super(SingleHeadAttention, self).__init__()
#         self.q_proj = nn.Linear(hidden_size, d_head)
#         self.k_proj = nn.Linear(hidden_size, d_head)
#         self.v_proj = nn.Linear(hidden_size, d_head)

#         self.dot_attention = ScaledDotProductAttention()

#     def forward(self, Q, K, V):
#         return self.dot_attention(self.q_proj(Q), self.k_proj(K), self.v_proj(V))
    
# class MultiHeadAttention(nn.Module):
#     def __init__(self,hidden_size, d_head, n_head):
#         super(MultiHeadAttention, self).__init__()

#         self.n_head = n_head

#         self.multi_head_attention = {}
#         for i in range(self.n_head):
#             head = SingleHeadAttention(hidden_size, d_head)
#             self.multi_head_attention[f"{i}_head"] = head

#     def forward(self, Q, K, V):
#         for i in range(self.n_head):
#             self.multi_head_attention[f"{i}_head"](Q, K, V) # batch x seq_len x d_head

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1) # specify the dimension to compute softmax

    def forward(self, Q, K, V, mask):
        # Q, K, V: batch x n_head x seq_len x d_head
        # mask: batch x seq_len
        d_k = K.size(-1)
        K_transpose = K.permute(0,1,3,2)

        attention = torch.matmul(Q,K_transpose) / math.sqrt(d_k) # batch x n_head x seq_len x seq_len
        masked_attetion = attention * mask

        softmax = self.softmax(attention)

        return torch.matmul(softmax, V) # batch x n_head x seq_len x d_head


class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_size, n_head, d_head):
        super(MultiHeadAttention, self).__init__()
        # (n_head x d_head = hidden_size)
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.d_head = d_head

        # Instead of n_head x 3 projections (hidden_size, d_head), 
        # define 3 integrated projections (hidden_size, hidden_size) and split them into each head
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.dot_attention = ScaledDotProductAttention()
        self.last_proj = nn.Linear(hidden_size, hidden_size)

    def mh_convert(self, proj):
        # batch x seq_len x hidden_size -> batch x n_head x seq_len x d_head
        mh_split = torch.split(proj, self.d_head, dim=-1) # n_head tensors (batch x seq_len x d_head)  
        return torch.stack(mh_split, dim=1) # batch x n_head x seq_len x d_head

    def forward(self, Q, K, V, mask):
        # linear projection
        q_projection = self.q_proj(Q)
        k_projection = self.k_proj(K)
        v_projection = self.v_proj(V)

        # multi head attention
        mh_attention = self.dot_attention(self.mh_convert(q_projection), 
                                          self.mh_convert(k_projection),
                                          self.mh_convert(v_projection),
                                          mask)
        
        # concat
        single_head = torch.split(mh_attention, 1, dim=1) # n_head tensors (batch x 1 x seq_len x d_head)
        concatenated = torch.cat(single_head, dim=-1).squeeze(1) # batch x seq_len x hidden_size

        return self.last_proj(concatenated) # batch x seq_len x hidden_size
    
class PositionWiseFF(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(PositionWiseFF, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # x: batch x seq_len x hidden_size
        x = self.ReLU(self.linear1(x))
        return self.linear2(x)
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, n_head, d_head, ff_size, dropout_prob):
        super(Encoder, self).__init__()
        self.mh_attention = MultiHeadAttention(hidden_size, n_head, d_head)
        
        self.ff_layer = PositionWiseFF(hidden_size, ff_size)

        self.dropout = nn.Dropout(dropout_prob)
        
        self.l_norm1 = nn.LayerNorm(hidden_size)
        self.l_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, src_mask):
        sublayer_mh = self.mh_attention(Q = x, K = x, V = x, mask = src_mask)
        sublayer_mh = self.dropout(sublayer_mh)
        x = self.l_norm1(x + sublayer_mh)

        sublayer_ff = self.ff_layer(x)
        sublayer_ff = self.dropout(sublayer_ff)
        return self.l_norm2(x + sublayer_ff)
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, n_head, d_head, ff_size, dropout_prob):
        super(Decoder, self).__init__()
        self.mh_attention1 = MultiHeadAttention(hidden_size, n_head, d_head)
        self.mh_attention2 = MultiHeadAttention(hidden_size, n_head, d_head)

        self.ff_layer = PositionWiseFF(hidden_size, ff_size)

        self.dropout = nn.Dropout(dropout_prob)

        self.l_norm1 = nn.LayerNorm(hidden_size)
        self.l_norm2 = nn.LayerNorm(hidden_size)
        self.l_norm3 = nn.LayerNorm(hidden_size)

    def forward(self, y):
        

        
class Transformer(nn.Module):
    def __init__(self, args, pad_idx):
        super(Transformer, self).__init__()
        
        self.vocab_size = args.word_count
        self.hidden_size = args.hidden_size
        self.pad_idx = pad_idx
        self.max_len = args.max_len

        # self.tg_embedding = nn.Embedding(num_embedding = self.vocab_size,
        #                              embedding_dim = self.hidden_size,
        #                              padding_idx = self.pad_idx)

    def forward(self, inputs):
        # inputs: batch x seq_len



        



                            

        