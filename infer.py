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


prompt = "Hope is the thing with feathers"
tokeniser = ByteLevelBPETokenizer()
encoded_prompt = tokeniser.encode(prompt).ids
gen_ids = generate_text(
    model, encoded_prompt, max_new_tokens=50, temperature=0.8, top_k=50
)
generated_text = tokeniser.decode(gen_ids)
print("Generated poem:\n", generated_text)
