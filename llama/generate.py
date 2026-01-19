import torch
from llama.model import LLaMA
from llama.train import CharDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# must match training config
C = 128
context_length = 64
T = 64

# load dataset for vocab
with open("../data/tinyshakespeare/input.txt") as f:
    text = f.read()
dataset = CharDataset(text, block_size=T)

# create model
model = LLaMA(
    n_layers=6,
    vocab_size=dataset.vocab_size,
    dim=C,
    context_length=context_length,
    num_head=4,
    num_kv_head=2,
).to(device)

# load checkpoint
model.load_state_dict(torch.load("checkpoints/epoch_7.pt", map_location=device))
model.eval()

# generate
prompt = "ROMEO:"
prompt_tokens = torch.tensor([dataset.encode(prompt)]).to(device)
print(prompt, end="", flush=True)
model.generate(
    prompt_tokens,
    max_new_tokens=20000,
    temperature=0.8,
    decode_fn=lambda i: dataset.itos[i],
)
