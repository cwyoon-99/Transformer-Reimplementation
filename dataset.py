import torch
from torch.utils.data import Dataset, DataLoader

class IwsltDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        item = self.dataset[idx]['translation']
        
        x = item['en']
        y = item['de']

        x = self.tokenizer.subword_tokenize(x)
        

        return x,y