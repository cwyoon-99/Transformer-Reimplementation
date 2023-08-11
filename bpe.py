import tqdm
import collections
import string
import logging

class BPE():
    def __init__(self, dataset, word_count):
        # pre-tokenize
        self.en_corpus = []
        for i in dataset:
            item = i['translation']
            pre_corpus = item['en'].split(" ") # split by space
            for j in pre_corpus:
                j = j + "</w>"  # add a special end token "</w>"
            self.en_corpus.extend(pre_corpus)

        self.pre_corpus_count = collections.Counter(self.en_corpus) # count the number of each element

        self.corpus_count = {}
        for k,v in self.pre_corpus_count.items():
            self.corpus_count[" ".join(list(k))] = v

        self.vocab = []
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
        self.vocab.append(freq_bigram)

        return freq_bigram

    def merge(self, freq_bigram):
        new_corpus_count = {}
        for k,v in self.corpus_count.items():
            new_key = k.replace(f"{freq_bigram}", f"{freq_bigram.replace(' ','')}")
            new_corpus_count[new_key] = v

        self.corpus_count = new_corpus_count

        # for k,v in self.corpus_count.items():
        #     if freq_bigram in k:
        #         print("yes")
        #         print(k)
        #         print(freq_bigram)

    def get_vocab(self):

        logging.info("Start BPE.")
        with open("merge.txt","w") as f:
            while self.word_count > len(self.vocab):
                freq_bigram = self.count_frequency() 
                f.write(f"{freq_bigram}\n")

                self.merge(freq_bigram)

        logging.info("finish BPE. save vocab.json...")

        with open("vocab.json","w") as f:
            for i in self.vocab:
                f.write(f"{i}\n")

    # tokenize an input based on merge info of BPE 
    def subword_tokenize(self, input):
        merge_info = []
        with open('merge.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                merge_info.append(line.strip())

        print(merge_info)
        # To find the longest subword merge, sort based on the length of each element
        merge_info = sorted(merge_info, key= len)

        words = input.split()
        for word in words:
            if len(word) == 1:
                continue
            else:
                word_split = list(word)

                # print(f"word split: \n{word_split}")
                
                bigrams = [" ".join(word_split[i:i+2]) for i in range(0, len(word_split) - 1)]

                # remove bigrams that are not in the merge.txt
                bigrams = [bigram for bigram in bigrams if bigram in merge_info]
                print(bigrams)

                # find the most longest sub-word merge
                priority_merge = max(bigrams, key = lambda x: merge_info.index(x))

                print(priority_merge)
                    
                
