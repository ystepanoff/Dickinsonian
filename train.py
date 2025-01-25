import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset


class DickinsonPoemsDataset(Dataset):
    def __init__(self, data_path, seq_len=64):
        tokeniser = ByteLevelBPETokenizer()
        tokeniser.train(
            files=data_path,
            vocab_size=2000,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
        )


dataset = DickinsonPoemsDataset("data/dickinson_clean.txt")
