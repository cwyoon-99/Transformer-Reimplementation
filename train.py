from datasets import load_dataset

import argparse
from bpe import BPE
from dataloader import IwsltDataset

parser = argparse.ArgumentParser()

parser.add_argument("--word_count", type= int, default = 37000)

args = parser.parse_args()


if __name__ == "__main__":

    # download and load the dataset
    dataset = load_dataset("iwslt2017", 'iwslt2017-en-de')

    # byte pair encoding
    bpe = BPE(dataset['train'], args.word_count)
    bpe.get_bigram()

    # train_dataset = IwsltDataset(dataset['train'])

    # print(train_dataset.__getitem__(5))
