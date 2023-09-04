from datasets import load_dataset

import argparse
import logging
import json
from tqdm import tqdm
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from utils import preprocess
from tokenizer import Tokenizer, TokenizerImport
from dataset import IwsltDataset, CustomCollate
from model import Transformer

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="transformer-base")
parser.add_argument("--word_count", type= int, default = 16000)
parser.add_argument("--src", type=str, default = "en")
parser.add_argument("--tg", type=str, default = "de")
parser.add_argument("--bpe_end_token", action="store_true", 
                    help= "whether a special end token </w> is added while training bpe")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--model_config_dir", type=str, default="model_config.json", 
                    help="configuration directory of model")
parser.add_argument("--train_config_dir", type=str, default="train_config.json",
                    help="configuration directory of training")
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--impl_bpe", action="store_true",
                    help="whether or not to select self-implemented BPE")

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

if __name__ == "__main__":
    # log
    logger = logging.getLogger('Transformers')
    logger.setLevel(logging.INFO)

    logging.basicConfig(level = logging.INFO, datefmt = '%Y-%m-%d %H%M%S'
                   ,format='%(asctime)s - %(message)s')

    # download and load the dataset
    logger.info("load dataset...")
    dataset = load_dataset("iwslt2017", 'iwslt2017-en-de', data_dir="data/")

    # preprocess
    train_data = preprocess(dataset['train'], args.src, args.tg)
    valid_data = preprocess(dataset['validation'], args.src, args.tg)
    test_data = preprocess(dataset['test'], args.src, args.tg)
    logger.info("load dataset complete")

    if args.impl_bpe:
        # create tokenizer using self-implmented BPE
        src_tokenizer = Tokenizer(train_data, args.src, args.word_count, args.bpe_end_token)
        tg_tokenizer = Tokenizer(train_data, args.tg, args.word_count, args.bpe_end_token)
    else:
        # create tokenizer from ByteLevelBPETokenizer
        src_tokenizer = TokenizerImport(train_data, args.src, args.word_count, args.bpe_end_token)
        tg_tokenizer = TokenizerImport(train_data, args.tg, args.word_count, args.bpe_end_token)

    # # BPE Test
    # src_test = "I have been blown away by this conference, and I want to thank all of you \
    # # for the many nice comments about what I had to say the other night."
    # print(f"src_test: {src_test.split()} \n src_bpe: {src_tokenizer.encode(src_test)}\n")

    # tg_test = "Ich bin wirklich begeistert von dieser Konferenz, und ich danke Ihnen allen \
    # für die vielen netten Kommentare zu meiner Rede vorgestern Abend."
    # print(f"tg_test: {tg_test.split()} \n tg_bpe: {tg_tokenizer.bpe_tokenize(tg_test)}\n")

    # load data
    # train_dataset = IwsltDataset(train_data[:1000], src_tokenizer, tg_tokenizer, args.src, args.tg)
    train_dataset = IwsltDataset(train_data, src_tokenizer, tg_tokenizer, args.src, args.tg)
    valid_dataset = IwsltDataset(valid_data, src_tokenizer, tg_tokenizer, args.src, args.tg)
    test_dataset = IwsltDataset(test_data, src_tokenizer, tg_tokenizer, args.src, args.tg)

    # pad_idx = src_tokenizer.stoi("<PAD>")
    pad_idx = src_tokenizer.stoi("<PAD>")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_workers,
                                   shuffle=True, collate_fn=CustomCollate(pad_idx=pad_idx, batch_first=True))
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers = args.num_workers,
                                   shuffle=False, collate_fn=CustomCollate(pad_idx=pad_idx, batch_first=True))
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers = args.num_workers,
                                   shuffle=False, collate_fn=CustomCollate(pad_idx=pad_idx, batch_first=True))

    # load model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer(hidden_size = args.hidden_size,
                        n_head = args.num_attention_heads,
                        d_head = args.num_hidden_size_per_head,
                        ff_size = args.intermediate_size,
                        dropout_prob= args.dropout_prob,
                        n_layer = args.num_stack_layers,
                        pad_idx = pad_idx,
                        vocab_size = args.word_count,
                        max_len = args.max_len,
                        lr = args.initial_lr,
                        adam_betas = (args.adam_b1, args.adam_b2),
                        adam_eps = args.adam_eps,
                        label_smooth_eps = args.label_smooth_eps,
                        warmup_steps = args.train_warmup,
                        tg_tokenizer = tg_tokenizer
                        )
    logger.info(f"Model Load Complete")

    # logger
    cur_time = time.strftime('%y%m%d_%H%M-')
    exp_name = f"{cur_time}_{args.model_name}"
    logger = CSVLogger(args.log_dir, name=exp_name)

    tb_logger = TensorBoardLogger("tb_logs", name=exp_name)
    # wandb_logger= WandbLogger("wandb_logs", name=exp_name)

    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="avg_val_pp",
        min_delta=0.00,
        patience=10,
        mode="min")

    checkpoint_callback = ModelCheckpoint(
            dirpath= os.path.join(args.log_dir, exp_name),
            filename="{epoch}_{avg_val_loss:.4f}_{avg_val_pp:.4f}_{avg_val_bleu:.4f}",
            save_top_k=2,
            monitor="avg_val_pp",
            mode="min",
        )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]

    # trainer
    trainer = pl.Trainer(max_epochs = args.num_epochs,
                         accelerator = "gpu",
                         devices = args.num_gpus,
                         strategy = "ddp_find_unused_parameters_false",
                        #  strategy = "ddp",
                         logger = [logger, tb_logger],
                        # logger = [logger, wandb_logger],
                         callbacks = callbacks,
                         )
    
    # train
    trainer.fit(model, train_loader, valid_loader)

    # test
    trainer.test(model, test_loader)
    