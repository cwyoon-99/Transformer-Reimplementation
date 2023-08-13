import tqdm
import collections
import string
import logging
import os

class BPE:
    def __init__(self, dataset, mode, word_count, bpe_end_token):
        # pre-tokenize
        self.corpus = []
        self.mode = mode
        self.word_count = word_count
        self.bpe_end_token = bpe_end_token

        self.save_dir = f"{self.mode}_bpe/"
        os.makedirs(self.save_dir, exist_ok=True)

        for item in dataset:
            pre_corpus = item[self.mode].split(" ") # split by space (pre_tokenize)

            self.corpus.extend(pre_corpus)

        self.pre_corpus_count = collections.Counter(self.corpus) # count the number of each element

        self.corpus_count = {}
        for k,v in self.pre_corpus_count.items():
            if self.bpe_end_token: # whether to add a special end token "</w>"
                self.corpus_count[" ".join(list(k)) + " </w>"] = v
            else:
                self.corpus_count[" ".join(list(k))] = v
        
        # merge info
        self.merge_info = []

        # vocab
        self.vocab = []
        # special token
        special_token = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]

        self.vocab.extend(special_token)

        # # base vocabulary
        # for word in self.corpus_count.keys():
        #     for letter in word:
        #         if letter not in self.vocab:
        #             self.vocab.append(letter)

    # get the most frequent bigram
    def count_frequency(self):        
        bigram_dict = {}
        for k,v in self.corpus_count.items():
            corpus_split = k.split()
            bigram_list = [" ".join(corpus_split[i:i+2]) for i in range(0, len(corpus_split) - 1)]

            for bigram in bigram_list:
                if bigram in bigram_dict:
                    bigram_dict[bigram] += v
                else:
                    bigram_dict[bigram] = v

        freq_bigram = max(bigram_dict, key = lambda x: bigram_dict[x])

        print(f"frequent bigram: {freq_bigram}")
        self.vocab.append(freq_bigram.replace(" ","")) # save vocab

        return freq_bigram

    def merge(self, freq_bigram):
        new_corpus_count = {}
        for k,v in self.corpus_count.items():
            # merge bigrams by removing a space
            new_key = k.replace(f"{freq_bigram}", f"{freq_bigram.replace(' ','')}")
            new_corpus_count[new_key] = v

        self.corpus_count = new_corpus_count

    def train(self):
        self.merge_dir = os.path.join(self.save_dir, "merge.txt")
        self.vocab_dir = os.path.join(self.save_dir, "vocab.json")

        if os.path.isfile(self.merge_dir):
            logging.info(f"Already have merge.txt. EXIT BPE training...")
            
            # load merge info
            with open(self.merge_dir, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    self.merge_info.append(line.strip())

            # load vocab
            with open(self.vocab_dir, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    self.vocab.append(line.strip())

        else:
            logging.info(f"Train {self.mode} BPE.")
            
            # train.
            with open(self.merge_dir,"w") as f:
                while self.word_count > len(self.vocab): # iterate until it exceeds the word limit
                    freq_bigram = self.count_frequency() # find the frequent bigram

                    self.merge(freq_bigram) # merge

                    self.merge_info.append(freq_bigram)
                    f.write(f"{freq_bigram}\n") # write merge

            logging.info(f"finish {self.mode} BPE...")

            # save vocab
            with open(self.vocab_dir,"w") as f:
                for i in self.vocab:
                    f.write(f"{i}\n")

        return self.merge_info, self.vocab

    # tokenize an input based on merge info of BPE
    def subword_tokenize(self, input, merge_info):
        tokenized_input = input.split() # pre-tokenize

        for idx in range(len(tokenized_input)):
            word = tokenized_input[idx]
            
            word_split = list(word)

            if self.bpe_end_token: # whether to add a special end token
                word_split.append("</w>")

            if len(word_split) <= 1:
                continue

            for merge in merge_info:
                i = 0
                while i < (len(word_split) - 1):
                    merge_bigram = " ".join(word_split[i:i+2]) 
                    if merge_bigram == merge:
                        word_split[i] = merge_bigram.replace(" ","")
                        del word_split[i + 1]
                    else:
                        i += 1
            
            # update
            tokenized_input[idx] = " ".join(word_split)

        tokenized_input = [sp for word in tokenized_input for sp in word.split()]

        return tokenized_input
                
