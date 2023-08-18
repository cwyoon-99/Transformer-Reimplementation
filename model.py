import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotAttention():

class SingleHeadAttention():

class MultiHeadAttention():

class PositionWiseFF():

class Encoder():

class Decoder():

class EncoderStack():

class DecoderStack():

class PositionalEncoding():

class Transformer(nn.Module):
    def __init__(self, args, pad_idx):
        super(Transformer, self).__init__()
        
        self.vocab_size = args.word_count
        self.hidden_size = args.hidden_size

        self.pad_idx = pad_idx
        
        self.src_embedding = nn.Embedding(num_embedding =self.vocab_size,
                                     embedding_dim = self.hidden_size,
                                     padding_idx = self.pad_idx)

        self.tg_embedding = nn.Embedding(num_embedding =self.vocab_size,
                                     embedding_dim = self.hidden_size,
                                     padding_idx = self.pad_idx)

        pe_function = lambda x, i : math.sin(x / (math.pow(10000, (2 * i) / self.hidden_size))) if 

        pos = torch.arange(start=0, end=self.hidden_size)

        

        self.positional_encoding = 

    def forward(self, inputs):
        # inputs: batch x seq_len

        input_embeddings = self.src_embedding(inputs['input_ids']) # batch x seq_len x hidden_size

        seq_len = input_embeddings.size(1)
        even_dimension = torch.arange(0, self.hidden_size, 2)
        odd_dimension = torch.arange(1, self.hidden_size, 2)

        positional_encoding = []
        for pos in range(seq_len):
            even

            freq = pos / 10000 ** ((2 * dimension) / self.hidden_size)

            torch.where()

            torch.where()



                            

        