import tqdm
import collections
import string

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

        self.corpus_count = collections.Counter(self.en_corpus) # count the number of each element
        self.vocab = [char for char in string.ascii_lowercase + string.ascii_uppercase] # default alphabets (lower and upper)
        self.word_count = word_count

    # get the most frequent bigram
    def get_bigram(self,):
        bigram_dict = {}
        for k,v in self.corpus_count.items():

            corpus_split = []
            corpus_split[0] = k

            vocab_list = []

            for vocab in reversed(self.vocab):
                for idx, corpus in enumerate(corpus_split):
                    if corpus in vocab_list:
                        continue

                    if vocab in corpus:
                        corpus_split[idx, idx + 1] = corpus.split(vocab)
                        vocab_list.append(vocab)         

            # split into bigram
            bigram_list = [" ".join(corpus_split[i:i+2]) for i in range(0, len(k)-1)]

            # merge
            for bigram in bigram_list:
                if bigram in bigram_dict:
                    bigram_dict[bigram] += v
                else:
                    bigram_dict[bigram] = v

        # add a frequent bigram
        self.vocab.append(max(bigram_dict, key = lambda x: bigram_dict[x]))

    # iterately get bigrams until vocab size does not exceed the word count
    def get_vocab(self):
        
        




        

