from bpe import BPE

# provide vocab
class Tokenizer:
    def __init__(self, dataset, mode, word_count, bpe_end_token):

        self.bpe = BPE(dataset, mode, word_count, bpe_end_token)

        # get merge_info and vocab
        self.merge_info, self.vocab = self.bpe.train()

    def bpe_tokenize(self, input):
        # subword tokenize
        tokenized_input = self.bpe.subword_tokenize(input, self.merge_info)

        return tokenized_input

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
            input_words.append(self.vocab[input_id])

        return input_words

