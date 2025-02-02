import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer


def generate_text_nucleus(model, encoded_prompt, tokenizer, max_new_tokens=60, temperature=0.8, top_p=0.9, device='cuda', max_seq_len=128):
    model.eval()
    model.to(device)

    input_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]

            logits = model(input_ids)[:, -1, :] / temperature

            # Apply nucleus (top-p) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask right to keep the first token above the threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id == tokenizer.token_to_id("\n"):
                print("newline")

            if next_token_id == tokenizer.token_to_id("<END>"):
                break

            generated_ids.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

    generated_text = tokenizer.decode(generated_ids).replace("<END>", "").strip()
    return generated_text


def generate_text(
    model,
    start_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    device="cuda",
):
    model.eval()
    model = model.to(device)

    if isinstance(start_ids, list):
        x = torch.tensor([start_ids], dtype=torch.long, device=device)
    else:
        x = start_ids.to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                _, indices = torch.topk(logits, top_k)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(1, indices, False)
                logits = logits.masked_fill(mask, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

    return x[0].tolist()

from main import LanguageModel

model = LanguageModel(
    vocab_size=7000,
    d_model=512,
    n_heads=8,
    n_layers=8,
    dim_feedforward=1024,
    max_seq_len=96,
    dropout=0.2,
)
model.load_state_dict(torch.load("checkpoints_4/dickinsonian.pth", weights_only=True))

prompt = "I dwell in Possibility..."
tokeniser = ByteLevelBPETokenizer()
tokeniser.train(
    files="data/dickinson_clean.txt",
    vocab_size=7000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>", "<eol>", "<END>"],
)
encoded_prompt = tokeniser.encode(prompt).ids
generated_text = generate_text_nucleus(model, encoded_prompt, tokeniser, max_new_tokens=60)
print("Generated poem:\n", generated_text)
