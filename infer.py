import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer


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
    vocab_size=2000,
    d_model=2048,
    n_heads=16,
    n_layers=16,
    dim_feedforward=256,
    max_seq_len=64,
    dropout=0.1,
)
model.load_state_dict(torch.load("checkpoints/dickinsonian.pth", weights_only=True))

prompt = "Come with me"
tokeniser = ByteLevelBPETokenizer()
tokeniser.train(
    files="data/dickinson_clean.txt",
    vocab_size=2000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>", "<eol>", "<END>"],
)
encoded_prompt = tokeniser.encode(prompt).ids
gen_ids = generate_text(
    model, encoded_prompt, max_new_tokens=60, temperature=0.5, top_k=50
)
generated_text = tokeniser.decode(gen_ids)
print("Generated poem:\n", generated_text)
