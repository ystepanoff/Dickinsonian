import math

import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset


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

    def __len__(self):
        return len(self.encoded_tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(
            self.encoded_tokens[idx : idx + self.seq_len],
            dtype=torch.long,
        )
        y = torch.tensor(
            self.encoded_tokens[idx + 1 : idx + self.seq_len + 1],
            dtype=torch.long,
        )
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len).unsqueeze(0)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
        self.scale = head_dim**-0.5

    def forward(self, x, mask=None):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        return torch.bmm(attn_weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(d_model, self.head_dim) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(n_heads * self.head_dim, d_model)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask=mask) for h in self.heads], dim=-1)
        out = self.linear(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dim_feedforward=1024,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, dim_feedforward, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embed(x)
        x = self.pos_encoding(x)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


batch_size = 32

dataset = DickinsonPoemsDataset("data/dickinson_clean.txt")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch_x, batch_y in dataloader:
    pass
