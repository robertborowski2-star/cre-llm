import torch
import torch.nn as nn
from model import TinyGPT, device, block_size, vocab_size

# ---- Load and encode the text ----
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocab (must match model.py)
chars  = sorted(set(text))
stoi   = { ch:i for i,ch in enumerate(chars) }
itos   = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode entire dataset as one long tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset size: {len(data):,} tokens")

# ---- Split into train / validation ----
# 90% train, 10% validation
# Validation lets us check if the model is learning or just memorizing
n      = int(0.9 * len(data))
train  = data[:n]
val    = data[n:]

# ---- Batch loader ----
# Grabs random chunks of text for training
def get_batch(split):
    d   = train if split == 'train' else val
    # pick random starting positions
    ix  = torch.randint(len(d) - block_size, (32,))
    # x = input sequence, y = same sequence shifted by 1 (the "answers")
    x   = torch.stack([d[i:i+block_size] for i in ix])
    y   = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---- Loss estimator ----
# Runs without gradient tracking (faster, saves memory)
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        L = torch.zeros(50)
        for k in range(50):
            x, y   = get_batch(split)
            logits = model(x)
            B,T,C  = logits.shape
            loss   = nn.functional.cross_entropy(
                logits.view(B*T, C),
                y.view(B*T)
            )
            L[k] = loss.item()
        losses[split] = L.mean()
    model.train()
    return losses

# ---- Training ----
model     = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Starting training...")
print(f"{'Step':>6}  {'Train Loss':>10}  {'Val Loss':>10}")
print("-" * 32)

for step in range(5000):

    # every 500 steps, print loss
    if step % 500 == 0:
        losses = estimate_loss(model)
        print(f"{step:>6}  {losses['train']:>10.4f}  {losses['val']:>10.4f}")

    # get a batch
    x, y   = get_batch('train')

    # forward pass
    logits = model(x)
    B,T,C  = logits.shape

    # calculate loss
    # cross_entropy measures how wrong our predictions were
    loss = nn.functional.cross_entropy(
        logits.view(B*T, C),  # flatten to (B*T, vocab)
        y.view(B*T)           # flatten targets
    )

    # backward pass — compute gradients
    optimizer.zero_grad()
    loss.backward()

    # update parameters
    optimizer.step()

# Final loss
losses = estimate_loss(model)
print(f"{'5000':>6}  {losses['train']:>10.4f}  {losses['val']:>10.4f}")

# Save the model weights
torch.save(model.state_dict(), 'model.pt')
print("\nModel saved to model.pt")

# ---- Generate some text ----
print("\n--- Generated text sample ---")
model.eval()
context = torch.zeros((1,1), dtype=torch.long).to(device)
generated = []
for _ in range(500):
    logits    = model(context[:, -block_size:])
    # take the last token's predictions
    logits    = logits[:, -1, :]
    # convert scores to probabilities
    probs     = torch.softmax(logits, dim=-1)
    # sample the next token
    next_tok  = torch.multinomial(probs, num_samples=1)
    context   = torch.cat([context, next_tok], dim=1)
    generated.append(next_tok.item())

print(decode(generated))
