""" Train the model on Tiny Shakespeare. Saves ckpt.pt, plus losses and samples along the way for the timelapse. """
import json
import torch
from model import GPT, block_size, load_data

batch_size = 32
max_iters = 5000
learning_rate = 1e-3
eval_every = 100
eval_iters = 50
sample_at = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(1337)
chars, train_data, val_data = load_data()
decode = lambda l: ''.join(chars[i] for i in l)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        out[split] = sum(model(*get_batch(split))[1].item() for _ in range(eval_iters)) / eval_iters
    model.train()
    return out


def sample():
    model.eval()
    torch.manual_seed(0)
    s = decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 300)[0].tolist())
    model.train()
    return s


model = GPT(len(chars)).to(device)
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device {device}')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
log = {'losses': [], 'samples': []}

for it in range(max_iters + 1):
    if it % eval_every == 0:
        losses = estimate_loss()
        log['losses'].append({'step': it, **losses})
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    if it in sample_at:
        rng = torch.get_rng_state()
        log['samples'].append({'step': it, 'text': sample()})
        torch.set_rng_state(rng)
    if it == max_iters:
        break
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({'model': model.state_dict(), 'chars': chars}, 'ckpt.pt')
json.dump(log, open('train_log.json', 'w'), indent=1)
print(log['samples'][-1]['text'])
