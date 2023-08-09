import torch
from torch.utils.data import Dataset, DataLoader

class IwsltDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        item = self.dataset[idx]['translation']
        
        x = item['en']
        y = item['de']

        return x,y