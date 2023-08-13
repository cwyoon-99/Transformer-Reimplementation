import torch
from torch.utils.data import Dataset, DataLoader

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
        src_ids = self.src_tokenizer.encode(src_words)

        
        
        return