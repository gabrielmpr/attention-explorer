""" Character-level GPT (Karpathy's "Let's build GPT" architecture), with two hooks for analysis:
every Head keeps its last attention matrix, and every MultiHeadAttention has a head mask for ablations. """
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters (same as v2_rebuild_1.py)
block_size = 64
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.wei = None # last attention matrix (B, T, T), for the explorer

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.wei = wei.detach()
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('head_mask', torch.ones(num_heads)) # set an entry to 0 to ablate that head

    def forward(self, x):
        out = torch.cat([h(x) * self.head_mask[i] for i, h in enumerate(self.heads)], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, return_residuals=False):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        residuals = [x]
        for block in self.blocks:
            x = block(x)
            residuals.append(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        if return_residuals:
            return logits, loss, residuals
        return logits, loss

    def attention(self):
        """ (n_layer, n_head, B, T, T) attention from the last forward pass """
        return torch.stack([torch.stack([h.wei for h in b.sa.heads]) for b in self.blocks])

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def load_data(path='input.txt'):
    text = open(path, 'r', encoding='utf-8').read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return chars, data[:n], data[n:]
