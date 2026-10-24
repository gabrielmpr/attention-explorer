""" Export ckpt.pt for the browser: docs/model.bin (all weights, float32, concatenated) + docs/model.json
(names, shapes, offsets, vocab), and docs/parity.json so the page can check its forward pass against PyTorch. """
import json
import numpy as np
import torch
from torch.nn import functional as F
from model import GPT, block_size, n_embd, n_head, n_layer

ckpt = torch.load('ckpt.pt', map_location='cpu')
chars = ckpt['chars']
model = GPT(len(chars))
model.load_state_dict(ckpt['model'])
model.eval()

tensors, offset, blobs = {}, 0, []
for name, t in model.state_dict().items():
    if name.endswith('tril') or name.endswith('head_mask'):
        continue  # rebuilt in JS
    a = t.detach().numpy().astype('<f4').ravel()
    tensors[name] = {'shape': list(t.shape), 'offset': offset}
    offset += a.size
    blobs.append(a)
np.concatenate(blobs).tofile('docs/model.bin')
json.dump({'chars': chars, 'block_size': block_size, 'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer,
           'tensors': tensors}, open('docs/model.json', 'w'))

# parity reference: one passage, plus the same passage with two heads switched off
text = 'ROMEO:\nBut soft, what light through yonder window breaks?'
idx = torch.tensor([[chars.index(c) for c in text]])
with torch.no_grad():
    full = F.softmax(model(idx)[0][0], -1)
    model.blocks[0].sa.head_mask[1] = 0
    model.blocks[2].sa.head_mask[3] = 0
    ablated = F.softmax(model(idx)[0][0], -1)
json.dump({'text': text, 'ablate': [[0, 1], [2, 3]],
           'probs_last': [round(v, 6) for v in full[-1].tolist()],
           'probs_last_ablated': [round(v, 6) for v in ablated[-1].tolist()]},
          open('docs/parity.json', 'w'))
print(f'{offset:,} floats, {offset * 4 / 1e6:.2f} MB')
