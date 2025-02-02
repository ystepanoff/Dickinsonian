import math
import os

import click
import torch
import torch.nn as nn
import torch.optim as optim
from fsspec.registry import default
from tokenizers import ByteLevelBPETokenizer
from torch.nn.functional import dropout
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
            special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>", "<eol>", "<END>"],
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
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
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
        d_model=128,
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


def train(model, dataloader, num_epochs=5, lr=3e-4, device="cuda", save_path="checkpoints/"):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            logits = logits.view(-1, logits.size(-1))
            targets = batch_y.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        save(model, save_path, epoch=epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def save(model, path, epoch=None):
    filename = "emily.pth"
    if epoch is not None:
        filename = f"emily.{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(path, filename))


@click.command()
@click.option("--data_path", type=click.Path(exists=True), help="Path to the data file", default="data/dickinson_clean.txt")
@click.option("--save_path", type=click.types.STRING, help="Path to save the model", default="checkpoints/")
@click.option("--vocab_size", type=click.types.INT, help="Vocabulary size", default=2000)
@click.option("--batch_size", type=click.types.INT, help="Batch size", default=32)
@click.option("--d_model", type=click.types.INT, help="Model dimension", default=256)
@click.option("--n_heads", type=click.types.INT, help="Number of heads", default=4)
@click.option("--n_layers", type=click.types.INT, help="Number of layers", default=4)
@click.option("--dim_feedforward", type=click.types.INT, help="Feedforward dimension", default=1024)
@click.option("--max_seq_len", type=click.types.INT, help="Maximum sequence length", default=64)
@click.option("--dropout", type=click.types.FLOAT, help="Dropout rate", default=0.1)
@click.option("--num_epochs", type=click.types.INT, help="Number of epochs", default=100)
@click.option("--learning_rate", type=click.types.FLOAT, help="Learning rate", default=3e-4)
@click.option("--device", type=click.types.STRING, help="Device to use", default="cuda")
def main(
    data_path,
    save_path,
    vocab_size,
    batch_size,
    d_model,
    n_heads,
    n_layers,
    dim_feedforward,
    max_seq_len,
    dropout,
    num_epochs,
    learning_rate,
    device,
):
    dataset = DickinsonPoemsDataset(data_path, seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    train(model, dataloader, num_epochs=num_epochs, lr=learning_rate, device=device, save_path=save_path)
    save(model, save_path)


if __name__ == "__main__":
    main()

