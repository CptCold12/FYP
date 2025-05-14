import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from torchtune.modules import RotaryPositionalEmbeddings

# === CONFIG ===
model_path = "timeline_model.pt"
token_map_path = "D:/CourseworkFolder/DPSynthData/Data Manipulation/token_map.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_new_tokens = 300
eos_token_id = 12665

with open(token_map_path, "r", encoding="utf-8") as f:
    token_map = json.load(f)
vocab_size = len(token_map)
itos = {v: k for k, v in token_map.items()}

def decode(token_ids):
    return [itos.get(i, f"<unk:{i}>") for i in token_ids]

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        assert n_embd % num_heads == 0, "Embedding size must be divisible by num_heads"

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=50000)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (batch_size, sequence_length, embedding_dim)

        q = self.query(x).view(B, T, self.num_heads, self.head_dim)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TimelineLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        min_tokens = 0
        eos_token = 12665

        for i in range(max_new_tokens):
            T = idx.size(1)
            tok_emb = self.token_embedding_table(idx)
            #pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = tok_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            probs = F.softmax(logits[:, -1, :], dim=-1)

            if i < min_tokens:
                probs[:, eos_token] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
         
            if next_token.item() == eos_token and i >= min_tokens:
                break

        return idx

model = TimelineLanguageModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
sequence = [55, 1637, 197, 338, 94, 7822, 32, 7976, 32, 7764, 173]

#context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
print("Generating synthetic timeline...")
output = model.generate(context, max_new_tokens=max_new_tokens)
decoded = decode(output[0].tolist())

print("Decoded Tokens:")
for token in decoded:
    print(token)

print("Full: [55, 1637, 197, 338, 94, 7822, 32, 7976, 32, 7764, 173, 7794, 197, 339, 197, 374, 0, 221, 22, 2353, 22, 2935, 12665]")
print("Input: [55, 1637, 197, 338, 94, 7822, 32, 7976, 32, 7764, 173]")
print(f"Output: {output[0].tolist()}")