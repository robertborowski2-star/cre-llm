import torch
import torch.nn as nn

# ---- Hyperparameters ----
# These are the "settings" of our model - we'll tune these later
batch_size  = 32    # how many sequences to train on in parallel
block_size  = 64    # how many characters the model sees at once (context window)
n_embd      = 64    # size of each token's embedding vector
n_head      = 4     # number of attention heads
n_layer     = 4     # number of transformer blocks stacked
dropout     = 0.1   # randomly zero out 10% of connections during training
vocab_size  = 62    # must match what tokenizer.py found

# ---- Device setup ----
# This is where M4 Metal kicks in
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# ---- Embedding Table ----
# Think of this as a lookup table: each of the 62 tokens
# gets its own row of 64 floats. The model learns what
# those floats should be during training.
class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb  = nn.Embedding(vocab_size, n_embd)  # character identity
        self.pos_emb    = nn.Embedding(block_size, n_embd)  # character position

    def forward(self, idx):
        B, T = idx.shape  # B = batch size, T = sequence length
        tok  = self.token_emb(idx)                             # (B, T, 64)
        pos  = self.pos_emb(torch.arange(T, device=device))   # (T, 64)
        return tok + pos   # combine token identity + position
    
# ---- Single Attention Head ----
# Each head learns a different "type" of relationship between tokens
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # mask: tokens can only look BACKWARDS, not at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # what do I contain?
        q = self.query(x)  # what am I looking for?

        # compute attention scores: how much does each token care about each other?
        wei = q @ k.transpose(-2,-1) * (C ** -0.5)  # (B, T, T)
        # mask out future tokens (can't cheat by looking ahead)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)  # normalize to probabilities
        wei = self.dropout(wei)

        v = self.value(x)     # what do I actually share?
        return wei @ v        # weighted sum of values

# ---- Multi-Head Attention ----
# Run several heads in parallel, each learning different relationships
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head  # 64 / 4 = 16 per head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run all heads, concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

# ---- Feed Forward ----
# After attention, each token processes itself independently
# This is where the model "thinks" about what it learned from attention
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # compress back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ---- Transformer Block ----
# One full layer: attention + feedforward + residual connections
# Residual connections (the += parts) help gradients flow during training
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # attention + residual
        x = x + self.ff(self.ln2(x))    # feedforward + residual
        return x

# ---- Full Model ----
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = TokenEmbedding()
        self.blocks    = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f      = nn.LayerNorm(n_embd)
        self.lm_head   = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        x = self.embedding(idx)   # (B, T, 64)
        x = self.blocks(x)        # (B, T, 64)
        x = self.ln_f(x)          # (B, T, 64)
        return self.lm_head(x)    # (B, T, 62) — one score per vocab token

# ---- Test the full model ----
model = TinyGPT().to(device)
dummy = torch.zeros((2, 8), dtype=torch.long).to(device)
logits = model(dummy)
print(f"Model output shape: {logits.shape}")  # expect (2, 8, 62)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
