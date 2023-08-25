import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class IwsltDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tg_tokenizer, src, tg):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tg_tokenizer = tg_tokenizer

        self.src = src
        self.tg = tg

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        item = self.dataset[idx]
        
        src_text = item[self.src]
        tg_text = item[self.tg]

        src_words = self.src_tokenizer.bpe_tokenize(src_text)
        src_words.insert(0, "<SOS>")
        src_words.append("<EOS>")
        src_ids = self.src_tokenizer.encode(src_words)

        tg_words = self.tg_tokenizer.bpe_tokenize(tg_text)
        # tg_words.insert(0, "<SOS>") # don't need to generate SOS
        tg_words.append("<EOS>")
        tg_ids = self.tg_tokenizer.encode(tg_words)
        
        return torch.tensor(src_ids), torch.tensor(tg_ids)
    
class CustomCollate:
    def __init__(self, pad_idx, batch_first):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        src_ids = [i[0] for i in batch]
        input_ids = pad_sequence(src_ids, batch_first=self.batch_first, padding_value=self.pad_idx)

        tg_ids = [i[1] for i in batch]
        decoder_input_ids = pad_sequence(tg_ids, batch_first=self.batch_first, padding_value=self.pad_idx)

        # mask (if token == pad -> 0, else -> 1)
        src_mask = torch.where(input_ids == self.pad_idx, 0, 1)

        tg_mask = torch.where(decoder_input_ids == self.pad_idx, 0, 1)

        return input_ids, decoder_input_ids, src_mask, tg_mask