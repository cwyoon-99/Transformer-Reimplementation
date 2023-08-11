from datasets import load_dataset

import argparse
from bpe import BPE
from dataset import IwsltDataset

parser = argparse.ArgumentParser()

parser.add_argument("--word_count", type= int, default = 37000)

args = parser.parse_args()


if __name__ == "__main__":

    # download and load the dataset
    dataset = load_dataset("iwslt2017", 'iwslt2017-en-de')

    # train bpe
    bpe = BPE(dataset['train'], args.word_count)

    bpe.subword_tokenize("I have been blown away by this conference, \
                         and I want to thank all of you for the many nice \
                          comments about what I had to say the other night.")

    # bpe.get_vocab()

    # train_dataset = IwsltDataset(dataset = dataset['train'], tokenizer = bpe)


    # print(train_dataset.__getitem__(5))
