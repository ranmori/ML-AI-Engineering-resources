This is the full script from the lecture — same pieces you already know (`Head`, `MultiHeadAttention`, `FeedForward`, the bigram skeleton), but now everything is wired together with new pieces: **residual connections**, **LayerNorm**, and a real training setup on actual text. Let's go top to bottom.

## Hyperparameters

```python
batch_size = 16     # sequences processed in parallel per step
block_size = 32      # max context length (how many past chars the model can see)
max_iters = 5000     # total training steps
eval_interval = 100  # how often to print loss
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200     # how many batches to average when measuring loss
n_embd = 64          # size of each token's embedding vector
n_head = 4           # number of attention heads per block
n_layer = 4          # number of transformer blocks stacked
dropout = 0.0
```

These are just settings, gathered at the top so you can tune the whole model in one place. `device` auto-detects GPU vs CPU — PyTorch doesn't do this for you; every tensor and model has to be explicitly `.to(device)`'d.

## Loading and encoding the text

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

- `text` is the raw training data (Tiny Shakespeare in the original lecture).
- `set(text)` finds every unique character used; `sorted(...)` puts them in a fixed order; `vocab_size` is how many unique characters exist (around 65 for Shakespeare: letters, punctuation, newline, etc.).
- `stoi` ("string to int") and `itos` ("int to string") are lookup dictionaries mapping each character to a number and back. Neural nets work with numbers, not letters, so every character gets an ID.
- `encode`/`decode` are one-liners using those dicts: text → list of integers, and back.

This is a **character-level** tokenizer — the crudest possible kind. Real models (GPT, etc.) use subword tokenizers that are far more efficient, but character-level keeps the lecture's vocab tiny and easy to reason about.

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
```

Encodes the entire text into one giant tensor of integers, then splits it 90/10. `train_data` is what the model learns from; `val_data` is held out to check whether the model is actually generalizing or just memorizing.

## `get_batch`: building training examples

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

- `ix = torch.randint(len(data) - block_size, (batch_size,))`: picks `batch_size` (16) random starting positions in the data. The `- block_size` ensures you never pick a start point so close to the end that there isn't enough text left to grab a full chunk.
- `x = torch.stack([data[i:i+block_size] for i in ix])`: for each random starting point, grab a chunk of `block_size` (32) consecutive characters. Stack all 16 chunks into one tensor, shape `(16, 32)`.
- `y` is the exact same thing, but shifted one character to the right: `data[i+1 : i+block_size+1]`.

This shift is the entire trick. At every position `t` in a chunk, `y[t]` is "the character that comes right after `x[t]`". So one 32-character chunk actually contains 32 separate training examples: predict character 2 from character 1, predict character 3 from characters 1–2, and so on, up to predicting character 33 from all 32 before it.

```python
x, y = x.to(device), y.to(device)
```
Moves both tensors to GPU if available — all the math has to happen on the same device.

## `estimate_loss`: measuring progress without training

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

- `@torch.no_grad()`: a decorator that disables gradient tracking for everything inside this function. We're only *measuring* loss here, not training, so there's no need to waste memory and compute storing gradients.
- `model.eval()`: switches the model into "evaluation mode." This matters because of `dropout` — in eval mode, dropout is turned off so you get a deterministic, stable loss reading. (Here `dropout = 0.0` so it has no visible effect, but the pattern is always there for when you turn dropout on.)
- For both the `'train'` and `'val'` splits, it grabs 200 random batches (`eval_iters`), computes the loss on each, and averages them. A single batch's loss is noisy (depends which random chunk you happened to sample); averaging 200 gives a much more stable number.
- `model.train()`: switches back to training mode before returning, so training can resume normally (with dropout active, if nonzero).
- Returns a dict: `{'train': ..., 'val': ...}`.

The reason both are reported: if train loss keeps dropping but val loss stalls or rises, the model is overfitting — memorizing training data rather than learning general patterns.

## `Head`

You've already seen this one. Two small differences from before:

```python
wei = q @ k.transpose(-2,-1) * C**-0.5
```
Earlier I wrote `head_size**-0.5`; here it's `C**-0.5`. Inside `forward`, `C` is reassigned from `x.shape` — but `x` here is the *output* of `self.key(x)` shape check... actually `B,T,C = x.shape` refers to the **input** `x`, where `C = n_embd = 64`. Wait — look closer: the scaling should really be by `head_size`, not the full embedding size. In this exact snippet, `C` in that line still refers to `x.shape`'s last dimension, i.e. `n_embd`, not `head_size`. This is a known quirk in the original lecture code — it's an approximation that still works fine in practice, but technically `head_size**-0.5` is the "correct" version from the paper. Don't worry about it; it doesn't break anything at this scale.

```python
self.dropout = nn.Dropout(dropout)
...
wei = self.dropout(wei)
```
New: after softmax, dropout randomly zeroes out some of the attention weights during training. This stops the model from relying too heavily on any single attention connection. Since `dropout = 0.0` here, it's inactive, but the hook is there for when you scale the model up and need more regularization.

## `MultiHeadAttention`

Same structure as before, with one addition:

```python
self.proj = nn.Linear(n_embd, n_embd)
...
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.dropout(self.proj(out))
```
Concatenating 4 heads of size 16 (`n_embd // n_head = 64 // 4 = 16`) gives back exactly `n_embd = 64`, so `proj` goes `64 → 64`. It's still there for the same reason as before — mixing the heads' outputs together — just with matching input/output size this time. Then dropout is applied to the projected output too.

## `FeedFoward`

Identical to what you already saw: expand to `4 * n_embd`, ReLU, project back down, dropout.

## `Block` — NEW: residual connections + LayerNorm

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

This bundles "communicate" (attention) and "compute" (feed-forward) into one reusable unit, which gets stacked multiple times to build depth.

**`head_size = n_embd // n_head`**: splits the embedding evenly across heads. `64 // 4 = 16` per head.

**LayerNorm** — `nn.LayerNorm(n_embd)`: for each token, normalizes its 64 numbers to have mean 0 and standard deviation 1 (then applies a small learnable scale/shift). Why bother? As you stack many layers, numbers flowing through the network can drift to very large or very small values, which makes training unstable and gradients unreliable. LayerNorm keeps every token's values in a consistent, well-behaved range at the start of every block, which makes deep networks much easier to train. It's applied per token, independently — it doesn't mix information across tokens or the batch, it just rescales each token's own vector.

**Residual connections** — `x = x + self.sa(self.ln1(x))`:

This is the single most important line for making deep networks trainable. Instead of replacing `x` with the attention output, you *add* the attention output to `x`. Same for the feed-forward step.

Why this matters: if attention learns nothing useful early in training (near-zero output), `x = x + ~0` just passes `x` through unchanged — information isn't lost, it's preserved by default. And during backpropagation, gradients can flow straight through the `+` back to earlier layers without being forced through every single transformation, which keeps gradients from vanishing in deep stacks. Without this `+`, stacking 4+ blocks would make training dramatically harder.

One subtlety: this is "pre-norm" — LayerNorm is applied *before* going into attention/feedforward (`self.sa(self.ln1(x))`), not after. This is a slight deviation from the original Transformer paper (which normalized after), but it's become the standard approach because it trains more stably.

**Shape check**: everything in and out of a `Block` is `(B, T, n_embd)`. Input and output shapes match exactly, which is exactly what lets you stack blocks back-to-back.

## `BigramLanguageModel` — the full model

```python
def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
```

Compare to the toy bigram model from before, where the embedding table directly output `vocab_size` logits. Now:

- `token_embedding_table`: `nn.Embedding(vocab_size, n_embd)` — each character ID maps to a 64-number vector (not directly to 65 logits anymore). This is a proper embedding now, not a shortcut lookup.
- `position_embedding_table`: `nn.Embedding(block_size, n_embd)` — a *separate* lookup table, with one 64-number vector per position (0 through 31). Attention has no built-in sense of order (it treats tokens as a set), so this gives the model a way to know "this is the 5th character in the sequence."
- `self.blocks = nn.Sequential(*[Block(...) for _ in range(n_layer)])`: builds 4 `Block` instances (`n_layer = 4`) and chains them with `nn.Sequential`, so calling `self.blocks(x)` runs `x` through block 1, feeds that output into block 2, and so on. The `*` unpacks the list into separate arguments, since `Sequential` expects each layer as its own argument, not a list.
- `ln_f`: one final LayerNorm after all the blocks, before the output layer. Common final touch for stability.
- `lm_head`: `nn.Linear(n_embd, vocab_size)` — "language model head," the final layer that turns each token's 64-number vector into 65 scores (logits), one per possible next character.

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)                          # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
    x = tok_emb + pos_emb                                              # (B,T,C)
    x = self.blocks(x)                                                 # (B,T,C)
    x = self.ln_f(x)                                                   # (B,T,C)
    logits = self.lm_head(x)                                           # (B,T,vocab_size)
```

- `tok_emb`: look up each character's embedding. Shape `(B, T, n_embd)`.
- `pos_emb`: `torch.arange(T, device=device)` makes `[0, 1, 2, ..., T-1]`, then looks up the position embedding for each. Shape `(T, n_embd)` — no batch dimension, since position 5 means the same thing regardless of which sequence in the batch you're in.
- `x = tok_emb + pos_emb`: broadcasting adds the `(T, n_embd)` position vectors onto every sequence in the `(B, T, n_embd)` batch. Now every token's vector encodes *both* "what character is this" and "where in the sequence is it."
- `x = self.blocks(x)`: runs through all 4 transformer blocks — communicate, think, communicate, think, repeated 4 times.
- `x = self.ln_f(x)`: final cleanup normalization.
- `logits = self.lm_head(x)`: project from 64 numbers per token up to 65 — the actual prediction.

The loss computation below this is identical to the toy bigram model you already walked through (reshape to `(B*T, C)`, `cross_entropy`).

## `generate` — one new detail

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond)
        ...
```

This is almost identical to before, with one addition: `idx_cond = idx[:, -block_size:]`. 

Why needed now: `position_embedding_table` only has entries for positions `0` through `block_size-1` (32 of them). As `idx` grows during generation (it can get arbitrarily long, token by token), feeding the *entire* sequence into the model would eventually ask for a position embedding beyond index 31 — which doesn't exist, and would crash. So before every forward pass, crop `idx` down to just the last `block_size` characters. The model only ever "sees" a sliding window of its 32 most recent characters, which matches `block_size` from training.

## The actual training loop

```python
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
```
Create the model, move it to GPU/CPU. `p.numel()` counts how many numbers are in each parameter tensor; summing over all parameters and dividing by 1e6 gives total parameter count in millions — just a sanity-check print.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

This is exactly the training loop from your very first question, just applied to the real model now: sample a batch, compute loss, clear old gradients, backpropagate, step the optimizer. The only new bit is the `if iter % eval_interval == 0` check — every 100 steps (and on the very last step), it pauses to print a proper averaged train/val loss using `estimate_loss()`, so you can watch the model actually improving over the full 5000 steps.

## Final generation

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```

Same pattern as before — start from a single token (ID 0), generate 2000 new characters, decode back to text. After 5000 training steps this should produce text that *looks* vaguely like Shakespeare (plausible words, sentence-like structure, "thou" and "lord" showing up) even though it's still a fairly small, character-level model and won't make real sense.

## The big picture

You now have the complete GPT architecture from this lecture:

```
characters → encode → embeddings (token + position)
  → [Block × 4]  (attention + feedforward, with residuals + LayerNorm, repeated)
  → final LayerNorm
  → linear layer → logits over vocabulary
  → cross-entropy loss (training) or softmax + sample (generation)
```

Everything you traced through earlier — `Head`, `MultiHeadAttention`, `FeedFoward` — are exactly the same classes, just now embedded inside `Block`, wrapped in residual connections and LayerNorm, stacked 4 deep, and trained on real text instead of toy tensors.
