# -*- coding: utf-8 -*-

from datasets import load_dataset

import argparse
import logging
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import preprocess
from tokenizer import Tokenizer
from dataset import IwsltDataset, CustomCollate
from model import Transformer

parser = argparse.ArgumentParser()

parser.add_argument("--word_count", type= int, default = 16000)
parser.add_argument("--src", type=str, default = "en")
parser.add_argument("--tg", type=str, default = "de")
parser.add_argument("--bpe_end_token", action="store_true", 
                    help= "whether a special end token </w> is added while training bpe")
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
combined_args.update(model_config)
combined_args.update(train_config)

args = argparse.Namespace(**combined_args)

def train_one_epoch(model, data_loader, optimizer, loss_fn):
    model.train(True)
    loss_interval = len(data_loader) // 10 # set the interval of loss

    for step, batch in enumerate(tqdm(data_loader)):
        running_loss = 0.
        last_loss = 0.

        inputs = {"input_ids": batch[0],
                "decoder_input_ids": batch[1],
                "src_mask": batch[2],
                "tg_mask": batch[3]
                }
        
        optimizer.zero_grad() # zero gradients for every batch
        
        outputs = model(inputs["input_ids"], 
                inputs["src_mask"],
                inputs["decoder_input_ids"][:,:-1], # exclude the last prediction (<EOS>)
                inputs["tg_mask"][:,:-1])
        
        loss = loss_fn(outputs, inputs["decoder_input_ids"][:,1:]) # exclude the first output (<SOS>)

        loss.backward()

        optimizer.step() # adjust learning weights

        # Gather data and report
        running_loss += loss.item()
        if step % loss_interval == loss_interval - 1:
            last_loss = running_loss / loss_interval # loss per batch
            logging.info('  batch {} loss: {}'.format(step + 1, last_loss))
            # tb_x = epoch_index * len(data_loader) + step + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_workers,
                                   shuffle=True, collate_fn=CustomCollate(pad_idx=pad_idx, batch_first=True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    model = Transformer(hidden_size = args.hidden_size,
                        n_head = args.num_attention_heads,
                        d_head = args.num_hidden_size_per_head,
                        ff_size = args.intermediate_size,
                        dropout_prob= args.dropout_prob,
                        n_layer = args.num_stack_layers,
                        pad_idx = pad_idx,
                        vocab_size = args.word_count,
                        max_len = args.max_len
                        ).to(device)
    
    logger.info(f"Model Load Complete")

    # optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=, betas=(args.adam_b1, args.adam_b2), eps=args.adam_eps)

    # learning rate scheduler (warmup_steps)
    schduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))

    # loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth_eps)

    logger.info(f"Start training ...")

    # for epoch in range(20):
    #     for input, target in dataset:
    #         optimizer.zero_grad()
    #         output = model(input)
    #         loss = loss_fn(output, target)
    #         loss.backward()
    #         optimizer.step()
    #     scheduler.step()
    