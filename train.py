# -*- coding: utf-8 -*-

from datasets import load_dataset

import argparse
import logging
import json

import torch
from torch.utils.data import DataLoader

from utils import preprocess
from tokenizer import Tokenizer
from dataset import IwsltDataset, CustomCollate

parser = argparse.ArgumentParser()

parser.add_argument("--word_count", type= int, default = 16000)
parser.add_argument("--src", type=str, default = "en")
parser.add_argument("--tg", type=str, default = "de")
parser.add_argument("--bpe_end_token", action="store_true", 
                    help= "whether a special end token </w> is added while training bpe")
parser.add_argument("--batch_first", action="store_false")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--model_config_dir", type=str, default="model_config.json", 
                    help="configuration directory of model")
parser.add_argument("--train_config_dir", type=str, default="train_config.json",
                    help="configuration directory of training")

args = parser.parse_args()

# load config file
with open(args.model_config_dir, 'r') as f:
    model_config = json.load(f)

with open(args.train_config_dir, 'r') as f:
    train_config = json.load(f)

# Combine the arguments
combined_args = vars(args)
combined_args.update(train_config)
combined_args.update(train_config)

args = argparse.Namespace(**combined_args)

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

    # preprocess
    train_data = preprocess(dataset['train'], args.src, args.tg)
    valid_data = preprocess(dataset['validation'], args.src, args.tg)
    test_data = preprocess(dataset['test'], args.src, args.tg)

    # create tokenizer using BPE
    src_tokenizer = Tokenizer(train_data, args.src, args.word_count, args.bpe_end_token)
    tg_tokenizer = Tokenizer(train_data, args.tg, args.word_count, args.bpe_end_token)

    # # BPE Test
    # src_test = "I have been blown away by this conference, and I want to thank all of you \
    # # for the many nice comments about what I had to say the other night."
    # print(f"src_test: {src_test.split()} \n src_bpe: {src_tokenizer.bpe_tokenize(src_test)}\n")

    # tg_test = "Ich bin wirklich begeistert von dieser Konferenz, und ich danke Ihnen allen \
    # für die vielen netten Kommentare zu meiner Rede vorgestern Abend."
    # print(f"tg_test: {tg_test.split()} \n tg_bpe: {tg_tokenizer.bpe_tokenize(tg_test)}\n")

    # load dataloader
    train_dataset = IwsltDataset(train_data, src_tokenizer, tg_tokenizer, args.src, args.tg)

    pad_idx = src_tokenizer.stoi("<PAD>")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_workers,
                                   shuffle=True, collate_fn=CustomCollate(pad_idx=pad_idx, batch_first=args.batch_first))
    
    # valid_dataloader = 
    # test_dataloader = 

    model = Transformer(args = args, pad_idx = pad_idx)

    # # optimizer (Adam)
    # optimizer = torch.optim.Adam(model.parameters(), lr=, betas=(), eps=,)

    # # learning rate scheduler (warmup_steps)
    # schduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))

    for step, batch in enumerate(train_dataloader):
        inputs = {"input_ids": batch[0],
                  "decoder_input_ids": batch[1],
                  "attention_mask": batch[2]
                  }

        break

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    # for epoch in range(20):
    #     for input, target in dataset:
    #         optimizer.zero_grad()
    #         output = model(input)
    #         loss = loss_fn(output, target)
    #         loss.backward()
    #         optimizer.step()
    #     scheduler.step()
    