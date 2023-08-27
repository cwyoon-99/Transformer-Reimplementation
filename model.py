import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
class
    1. Embedding parts
        PositionalEncoding
        WordEmbedding
    2. Multi-Head Attention (sublayer 1)
        ScaledDotProductAttention
        MultiHeadAttention
    3. FF (sublayer 2)
        PositionWiseFF
    4. Model
        Encoder
        Decoder
        EncoderStack
        DecoderStack
        Transformer
"""

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super(PositionalEncoding, self).__init__()

        even_index = torch.arange(0, hidden_size, 2) # [0,2,4,6, ...]
        _partial = 10000 ** (even_index / hidden_size)

        self.positional_encoding = torch.zeros(max_len, hidden_size) # seq_len x hidden_size
        self.positional_encoding.requires_grad = False # positional encoding is not training parameters

        for pos in range(max_len):
            self.positional_encoding[pos,0::2] = torch.sin(pos / _partial)
            self.positional_encoding[pos,1::2] = torch.cos(pos / _partial)

    def forward(self, x):
        # x: batch x seq_len 
        seq_len = x.size(1)
        return self.positional_encoding[:seq_len, :].unsqueeze(0) # 1 x seq_len x hidden_size

class WordEmbedding(nn.Module):
    def __init__(self, pad_idx, vocab_size, max_len, hidden_size):
        super(WordEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings = vocab_size,
                                     embedding_dim = hidden_size,
                                     padding_idx = pad_idx)
        
        self.positional_encoding = PositionalEncoding(max_len, hidden_size)

    def forward(self, x):
        # x: batch  x seq_len
        w_embedding = self.word_embedding(x) # batch x seq_len x hidden_size
        return w_embedding + self.positional_encoding(x) # add positional encoding to input embedding
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1) # specify the dimension to compute softmax

    def forward(self, Q, K, V, pad_mask, attn_mask):
        # Q, K, V: batch x n_head x len x d_head
        # pad_mask: batch x len
        # attn_mask: batch x tg_len x tg_len
        batch, _, q_len, _ = Q.size()
        _, _, k_len, d_k = K.size()

        K_transpose = K.permute(0,1,3,2)

        attention = torch.matmul(Q,K_transpose) / math.sqrt(d_k) # batch x n_head x len x len

        mask = torch.ones(batch, q_len, k_len)

        # <PAD> masking
        if pad_mask is not None:
            expanded_pad_mask = pad_mask.unsqueeze(2) * pad_mask.unsqueeze(1) # batch x len x len
            mask = mask * expanded_pad_mask
        
        # masked MH masking
        if attn_mask is not None:
            mask = mask * attn_mask

        # print(mask.shape)
        # print(attention.shape)

        attention = torch.where(mask.unsqueeze(1) != 0, attention, torch.tensor(-1e-9, dtype=torch.float32))

        softmax = self.softmax(attention)

        return torch.matmul(softmax, V) # batch x n_head x len x d_head

class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_size, n_head, d_head):
        super(MultiHeadAttention, self).__init__()
        # (n_head x d_head = hidden_size)
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.d_head = d_head

        # Instead of n_head x 3 projections, 
        # define 3 integrated projections (hidden_size, hidden_size) and split them
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.dot_attention = ScaledDotProductAttention()
        self.last_proj = nn.Linear(hidden_size, hidden_size)

    def mh_convert(self, proj):
        # batch x seq_len x hidden_size -> batch x n_head x seq_len x d_head
        mh_split = torch.split(proj, self.d_head, dim=-1) # n_head tensors (batch x seq_len x d_head)
        return torch.stack(mh_split, dim=1) # batch x n_head x seq_len x d_head

    def forward(self, Q, K, V, pad_mask = None, attn_mask = None):
        # linear projection
        q_projection = self.q_proj(Q)
        k_projection = self.k_proj(K)
        v_projection = self.v_proj(V)

        # multi head attention
        mh_attention = self.dot_attention(self.mh_convert(q_projection), 
                                          self.mh_convert(k_projection),
                                          self.mh_convert(v_projection),
                                          pad_mask = pad_mask,
                                          attn_mask = attn_mask)
        
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
        # encoder self attention
        sublayer_mh = self.mh_attention(Q = x, K = x, V = x, pad_mask  = src_mask)
        sublayer_mh = self.dropout(sublayer_mh)
        x = self.l_norm1(x + sublayer_mh)

        # feed-forward
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

    def forward(self, x, src_mask, y, tg_mask):
        batch, tg_len, _ = y.size()

        # Replace with torch.tril()
        # output_mask = (torch.eye(tg_len)).repeat(batch,1,1,1) # batch x 1 x tg_len x tg_len
        # for i in range(tg_len):
        #     output_mask[:,:,i,:i] = 1 # lower triangle part

        output_mask = torch.ones(batch, tg_len, tg_len)
        output_mask = torch.tril(output_mask) # batch x tg_len x tg_len
        
        # decoder masked attention
        sublayer_mmh = self.mh_attention1(Q = y, K = y, V = y, pad_mask = tg_mask, attn_mask = output_mask)
        sublayer_mmh = self.dropout(sublayer_mmh)
        y = self.l_norm1(y + sublayer_mmh)

        # encoder-decoder attention
        sublayer_ed_mh = self.mh_attention2(Q = y, K = x, V = x)
        # sublayer_ed_mh = self.mh_attention2(Q = y, K = x, V = x, pad_mask = src_mask) # pad_mask = src_mask?
        sublayer_ed_mh = self.dropout(sublayer_ed_mh)
        y = self.l_norm2(y + sublayer_ed_mh)

        # feed-forward
        sublayer_ff = self.ff_layer(y)
        sublayer_ff = self.dropout(sublayer_ff)
        return self.l_norm3(y + sublayer_ff)

class EncoderStack(nn.Module):
    def __init__(self, hidden_size, n_head, d_head, ff_size, dropout_prob, n_layer, 
                 pad_idx, vocab_size, max_len):
        super(EncoderStack, self).__init__()

        self.input_embedding = WordEmbedding(pad_idx, vocab_size, max_len, hidden_size)

        encoder_list = []
        for i in range(n_layer):
            encoder_list.append(Encoder(hidden_size, n_head, d_head, ff_size, dropout_prob))

        self.encoder_stack = nn.ModuleList(encoder_list)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, src_mask):
        x = self.input_embedding(x)
        x = self.dropout(x)

        for stack in self.encoder_stack:
            x = stack(x, src_mask)

        return x
    
class DecoderStack(nn.Module):
    def __init__(self, hidden_size, n_head, d_head, ff_size, dropout_prob, n_layer,
                 pad_idx, vocab_size, max_len):
        super(DecoderStack, self).__init__()

        self.output_embedding = WordEmbedding(pad_idx, vocab_size, max_len, hidden_size)

        decoder_list = []
        for i in range(n_layer):
            decoder_list.append(Decoder(hidden_size, n_head, d_head, ff_size, dropout_prob))

        self.decoder_stack = nn.ModuleList(decoder_list)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, src_mask, y, tg_mask):
        y = self.output_embedding(y)
        y = self.dropout(y)

        for stack in self.decoder_stack:
            y = stack(x,src_mask, y, tg_mask)

        return y
        
class Transformer(nn.Module):
    def __init__(self, hidden_size, n_head, d_head, ff_size, dropout_prob, n_layer, 
                 pad_idx, vocab_size, max_len):
        super(Transformer, self).__init__()

        self.encoders = EncoderStack(hidden_size, n_head, d_head, ff_size, dropout_prob, n_layer, 
                 pad_idx, vocab_size, max_len)
        
        self.decoders = DecoderStack(hidden_size, n_head, d_head, ff_size, dropout_prob, n_layer, 
                 pad_idx, vocab_size, max_len)
        
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, src_mask, output_ids, tg_mask):
        # inputs: batch x seq_len
        # src_mask: batch x seq_len
        # outputs: batch x tg_len
        # tg_mask: batch x tg_len
        x = self.encoders(input_ids, src_mask) # batch x seq_len x hidden_size
        y = self.decoders(x, src_mask, output_ids, tg_mask) # batch x tg_len x hidden_size
        y = self.linear(y) # batch x tg_len x vocab_size
        return self.softmax(y) # batch x tg_len x vocab_size