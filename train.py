import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset


class DickinsonPoemsDataset(Dataset):
    def __init__(self, data_path, seq_len=64):
        self.encoded_tokens = []
        self.seq_len = seq_len

        tokeniser = ByteLevelBPETokenizer()
        tokeniser.train(
            files=data_path,
            vocab_size=2000,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>", "<eol>"],
        )

        with open(data_path, "r") as data_file:
            for line in data_file:
                line = line.strip("\n")
                encoding = tokeniser.encode(line)
                self.encoded_tokens.extend(encoding.ids)
            self.encoded_tokens.append(tokeniser.token_to_id("<eol>"))


dataset = DickinsonPoemsDataset("data/dickinson_clean.txt")
print(dataset.encoded_tokens)
