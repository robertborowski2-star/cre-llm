# Read the text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# --- Step 1: Build the vocabulary ---
# Get every unique character in the text
chars = sorted(set(text))
vocab_size = len(chars)

print(f"Total characters in text: {len(text):,}")
print(f"Unique characters (vocabulary size): {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

# --- Step 2: Build encoder and decoder ---
# Encoder: character → number
stoi = { ch:i for i,ch in enumerate(chars) }
# Decoder: number → character
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]  # string → list of integers

def decode(l):
    return ''.join([itos[i] for i in l])  # list of integers → string

# --- Step 3: Test it ---
test = "Hello!"
encoded = encode(test)
decoded = decode(encoded)

print(f"\nTest string: '{test}'")
print(f"Encoded: {encoded}")
print(f"Decoded back: '{decoded}'")
