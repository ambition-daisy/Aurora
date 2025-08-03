from transformers import PreTrainedTokenizer, AutoTokenizer
import os

class AminoAcidTokenizer(PreTrainedTokenizer):
    def __init__(self, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]", **kwargs):
        vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "A": 4,
            "C": 5,
            "D": 6,
            "E": 7,
            "F": 8,
            "G": 9,
            "H": 10,
            "I": 11,
            "K": 12,
            "L": 13,
            "M": 14,
            "N": 15,
            "P": 16,
            "Q": 17,
            "R": 18,
            "S": 19,
            "T": 20,
            "V": 21,
            "W": 22,
            "Y": 23,
            "X": 24,
            "[AG_START]":25,
            "[AG_END]":26,
            "[AB_START]":27,
            "[AB_END]":28
        }
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        super().__init__(pad_token=pad_token, unk_token=unk_token, eos_token=eos_token, bos_token=bos_token, **kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = os.path.join(save_directory, (filename_prefix or "") + "vocab.txt")
        with open(path, "w") as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(token + "\n")
        return (path,)


