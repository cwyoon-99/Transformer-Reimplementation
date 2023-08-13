from datasets import load_dataset

import argparse
import logging

from utils import preprocess
from tokenizer import Tokenizer
from dataset import IwsltDataset

parser = argparse.ArgumentParser()

parser.add_argument("--word_count", type= int, default = 37000)
parser.add_argument("--src", type=str, default = "en")
parser.add_argument("--tg", type=str, default = "de")
parser.add_argument("--bpe_end_token", action="store_true", help= "whether a special end token </w> is added while training bpe")

args = parser.parse_args()


if __name__ == "__main__":

    # log
    logger = logging.getLogger('Transformers')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logging.basicConfig(level = logging.INFO, datefmt = '%Y-%m-%d %H%M%S'
                   ,format='%(asctime)s - %(message)s')

    # download and load the dataset
    dataset = load_dataset("iwslt2017", 'iwslt2017-en-de')

    train_dataset = preprocess(dataset['train'], args.src, args.tg)
    valid_dataset = preprocess(dataset['validation'], args.src, args.tg)
    test_dataaset = preprocess(dataset['test'], args.src, args.tg)

    src_tokenizer= Tokenizer(train_dataset, args.src, args.word_count, args.bpe_end_token)
    tg_tokenizer = Tokenizer(train_dataset, args.tg, args.word_count, args.bpe_end_token)

    # IwsltDataset(train_dataset, src_tokenizer, tg_tokenizer, args.src, args.tg)