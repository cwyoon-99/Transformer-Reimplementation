import os

from bpe import BPE
from tokenizers import ByteLevelBPETokenizer

class Tokenizer:
    def __init__(self, dataset, mode, word_count, bpe_end_token):

        self.bpe = BPE(dataset, mode, word_count, bpe_end_token)

        # get merge_info and vocab
        self.merge_info, self.vocab = self.bpe.train()

    def bpe_tokenize(self, input):
        # subword tokenize
        tokenized_input = self.bpe.subword_tokenize(input, self.merge_info)

        return tokenized_input
    
    def stoi(self, string):
        if string in self.vocab:
            return self.vocab.index(string)
        else:
            return self.vocab.index("[UNK]")

    def itos(self, idx):
        return self.vocab[idx]

    # Encode tokenized inputs into input_ids
    def encode(self, input_words):
        # map ids
        input_ids = []
        for token in input_words:
            if token in self.vocab:
                input_id = self.vocab.index(token)
            else:
                input_id = self.vocab.index("[UNK]")

            input_ids.append(input_id) # assign input_id

        return input_ids
    
    # Decode
    def decode(self, input_ids):
        input_words = []
        for input_id in input_ids:
            input_words.append(self.itos(input_id))

        return input_words

class TokenizerImport:
    def __init__(self, dataset, mode, word_count, bpe_end_token):

        self.bpe = ByteLevelBPETokenizer()

        texts = " ".join([item[mode] for item in dataset])

        # save in txt file
        os.makedirs("texts/", exist_ok = True)
        text_dir = f"texts/iwslt2017_{mode}.txt"
        if not os.path.isfile(text_dir):
            with open(text_dir,"w") as f:
                f.write(texts)

        # train
        save_dir = f"{mode}_bpe_import"
        os.makedirs(save_dir, exist_ok = True)
        special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]
        if not os.path.isfile(os.path.join(save_dir, "merges.txt")):
            self.bpe.train(files = text_dir, vocab_size=word_count,
                           min_frequency=2, special_tokens=special_tokens)
            self.bpe.save_model(save_dir)

        self.bpe = ByteLevelBPETokenizer(
            os.path.join(save_dir, "vocab.json"),
            os.path.join(save_dir, "merges.txt")
        )
        
        self.bpe.add_special_tokens(special_tokens)

    def bpe_tokenize(self, input):
        return self.bpe.encode(input).tokens
    
    def stoi(self, string):
        return self.bpe.token_to_id(string)

    def encode(self, input_words):
        return self.bpe.encode(input_words).ids
    
    def decode(self, input_ids):
        return self.bpe.decode(input_ids, skip_special_tokens=True)