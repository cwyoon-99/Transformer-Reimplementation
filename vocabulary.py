from bpe import BPE

class Vocabulary:
    def __init__(self, dataset, word_count):
        self.dataset = dataset


        self.src_bpe = BPE(dataset['train'], "src", word_count)
        self.tg_bpe = BPE(dataset['train'], "tg", word_count)

        


    # map tokenized inputs into input_ids
    def numericalize(self, )