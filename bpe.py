import tqdm
import collections
import string
import logging
import os

class BPE:
    def __init__(self, dataset, mode, word_count):
        # pre-tokenize
        self.corpus = []
        self.lang = "en" if mode == "src" else "de"

        self.save_dir = f"{self.lang}_bpe/"
        os.makedirs(self.save_dir, exist_ok=True)

        for i in dataset:
            item = i['translation']
            pre_corpus = item[self.lang].split(" ") # split by space (pre_tokenize)
            for j in pre_corpus:
                j = j + "</w>"  # add a special end token "</w>"
            self.corpus.extend(pre_corpus)

        self.pre_corpus_count = collections.Counter(self.corpus) # count the number of each element

        self.corpus_count = {}
        for k,v in self.pre_corpus_count.items():
            self.corpus_count[" ".join(list(k))] = v

        # define special token
        special_token = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        self.vocab = []
        self.vocab.append(special_token)

        # base vocabulary
        for word in self.corpus_count.keys():
            for letter in word:
                if letter not in self.vocab:
                    self.vocab.append(letter)

        self.word_count = word_count


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
        self.vocab.append(freq_bigram.replace(" ",""))

        return freq_bigram


    def merge(self, freq_bigram):
        new_corpus_count = {}
        for k,v in self.corpus_count.items():
            new_key = k.replace(f"{freq_bigram}", f"{freq_bigram.replace(' ','')}")
            new_corpus_count[new_key] = v

        self.corpus_count = new_corpus_count


    def get_vocab(self):
        self.merge_dir = os.path.join(self.save_dir, "merge_txt")
        if os.path.isfile(self.merge_dir):
            logging.info(f"Already have merge.txt. EXIT BPE training...")
            return
        else:
            logging.info(f"Start {self.lang} BPE.")
            with open(self.merge_dir,"w") as f:
                while self.word_count > len(self.vocab):
                    freq_bigram = self.count_frequency() # find the frequent bigram 

                    self.merge(freq_bigram) # merge it

                    f.write(f"{freq_bigram}\n") # write merge info

            logging.info(f"finish {self.lang} BPE. save vocab.json...")

            self.vocab_dir = os.path.join(self.save_dir, "vocab.json")
            # write vocab
            with open(self.vocab_dir,"w") as f:
                for i in self.vocab:
                    f.write(f"{i}\n")


    # tokenize an input based on merge info of BPE
    def subword_tokenize(self, input):
        merge_info = []

        # load merge info
        with open(self.merge_dir, 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                merge_info.append(line.strip())

        tokenized_input = input.split() # pre-tokenize

        for idx in range(len(tokenized_input)):
            word = tokenized_input[idx]
            if len(word) == 1:
                continue
            else:
                word_split = list(word)

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
                
