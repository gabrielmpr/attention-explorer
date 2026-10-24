""" Everything that reads ckpt.pt: head ablations, attention-by-offset, logit lens, embeddings, the training
timelapse, and docs/data.json for the explorer page. Runs on CPU in about a minute. """
import json
import os
import sys
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
import style
from model import GPT, block_size, n_layer, n_head, load_data

THEME = sys.argv[1] if len(sys.argv) > 1 else 'dark'  # figures for the page's dark or light theme
SFX = '' if THEME == 'dark' else '_light'
style.use(THEME)
os.makedirs('docs/figures', exist_ok=True)
os.makedirs('docs', exist_ok=True)
torch.manual_seed(0)

ckpt = torch.load('ckpt.pt', map_location='cpu')
chars, _, val_data = load_data()
assert chars == ckpt['chars']
model = GPT(len(chars))
model.load_state_dict(ckpt['model'])
model.eval()
show = lambda c: {'\n': '⏎', ' ': '·'}.get(c, c)

# a fixed evaluation set: 100 batches of 32 windows from the validation split
ix = torch.randint(len(val_data) - block_size, (3200,))
X = torch.stack([val_data[i: i + block_size] for i in ix])
Y = torch.stack([val_data[i + 1: i + block_size + 1] for i in ix])


@torch.no_grad()
def val_loss():
    return np.mean([model(X[i:i + 32], Y[i:i + 32])[1].item() for i in range(0, len(X), 32)])


# ---------- 1. head ablation: zero one head, measure how much val loss gets worse ----------
base = val_loss()
delta = np.zeros((n_layer, n_head))
for l in range(n_layer):
    for h in range(n_head):
        model.blocks[l].sa.head_mask[h] = 0
        delta[l, h] = val_loss() - base
        model.blocks[l].sa.head_mask[h] = 1
print(f'val loss {base:.4f}')
print('ablation Δloss (rows = layers, cols = heads):\n', np.round(delta, 3))

# ---------- 2. what each head looks at: average attention by offset (t - j) ----------
max_off = 16
with torch.no_grad():
    model(X[:512])
att = model.attention()  # (L, H, B, T, T)
T = block_size
off = torch.arange(T)[:, None] - torch.arange(T)[None, :]  # off[t, j] = t - j
profile = np.zeros((n_layer, n_head, max_off))
for o in range(max_off):
    m = (off == o)
    profile[:, :, o] = (att[..., m].sum(-1) / m.sum()).mean(-1).numpy()  # mean over valid t and over batch

# mass on spaces/newlines, a hint for "word boundary" heads
xb = X[:512]
is_space = ((xb == chars.index(' ')) | (xb == chars.index('\n'))).float()  # (B, T)
space_mass = (att * is_space[None, None, :, None, :]).sum(-1).mean((-1, -2)).numpy()
mean_dist = (att * off.clamp(min=0).float()).sum(-1).mean((-1, -2)).numpy()

heads = []
for l in range(n_layer):
    for h in range(n_head):
        heads.append({'layer': l, 'head': h, 'delta': round(float(delta[l, h]), 4),
                      'self': round(float(profile[l, h, 0]), 3), 'prev': round(float(profile[l, h, 1]), 3),
                      'space': round(float(space_mass[l, h]), 3), 'dist': round(float(mean_dist[l, h]), 2),
                      'profile': [round(float(p), 3) for p in profile[l, h]]})
for r in heads:
    print(f"L{r['layer']}H{r['head']}  Δ{r['delta']:+.3f}  self {r['self']:.2f}  prev {r['prev']:.2f}  space {r['space']:.2f}  dist {r['dist']:.1f}")

# single heads barely matter, so switch off groups: whole layers, and all the previous-character heads at once
def ink_on(rgba):
    ''' dark or light text, whichever reads on this cell color '''
    return '#111111' if 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.5 else '#ffffff'


def ablate(group):
    for l, h in group:
        model.blocks[l].sa.head_mask[h] = 0
    d = val_loss() - base
    for l, h in group:
        model.blocks[l].sa.head_mask[h] = 1
    return d

prev_heads = [(r['layer'], r['head']) for r in heads if r['prev'] > 0.6]
fmt = lambda g: ', '.join(f'L{l}H{h}' for l, h in g)
groups = [(f'every head in layer {l}', [(l, h) for h in range(n_head)]) for l in range(n_layer)]
groups += [(f'previous-char heads, one at a time (max)', None),
           (f'all previous-char heads together\n({fmt(prev_heads)})', prev_heads),
           ('every attention head', [(l, h) for l in range(n_layer) for h in range(n_head)])]
group_delta = []
for name, g in groups:
    d = max(delta[l, h] for l, h in prev_heads) if g is None else ablate(g)
    group_delta.append({'name': name, 'delta': round(float(d), 4)})
    print(f'{name.splitlines()[0]:45s} Δ{d:+.3f}')

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.25]})
vmax = delta.max()
im = ax.imshow(delta, cmap=style.SEQ, vmin=0, vmax=vmax)
for l in range(n_layer):
    for h in range(n_head):
        ax.text(h, l, f'+{delta[l, h]:.3f}', ha='center', va='center', fontsize=12,
                color=ink_on(plt.get_cmap(style.SEQ)(delta[l, h] / vmax)))
ax.set_xticks(range(n_head), [f'head {h}' for h in range(n_head)])
ax.set_yticks(range(n_layer), [f'layer {l}' for l in range(n_layer)])
ax.set_title(f'Val loss increase, one head switched off\n(baseline {base:.3f})', loc='left')
for s in ax.spines.values():
    s.set_visible(False)
names_g = [g['name'] for g in group_delta][::-1]
vals_g = [g['delta'] for g in group_delta][::-1]
ax2.barh(range(len(vals_g)), vals_g, color=[style.ACCENTS[0] if 'together' in n else style.ACCENTS[2] for n in names_g])
ax2.set_yticks(range(len(vals_g)), names_g, fontsize=10)
ax2.set_xscale('log')
for i, v in enumerate(vals_g):
    ax2.text(v, i, f'  +{v:.3f}', va='center', fontsize=10)
ax2.set_title('Val loss increase, groups switched off (log scale)', loc='left')
ax2.grid(axis='x', color=style.GRID)
fig.tight_layout()
fig.savefig(f'docs/figures/ablation{SFX}.png', dpi=style.DPI)

fig, axes = plt.subplots(n_layer, n_head, figsize=(12, 9), sharex=True, sharey=True)
for l in range(n_layer):
    for h in range(n_head):
        a = axes[l, h]
        a.bar(range(max_off), profile[l, h], color=style.ACCENTS[2], width=0.8)
        a.set_title(f'L{l} H{h}   Δloss +{delta[l, h]:.3f}', fontsize=10, loc='left')
        a.grid(axis='y', color=style.GRID)
for a in axes[-1]:
    a.set_xlabel('how far back (0 = itself)')
fig.suptitle('Where each head looks: average attention by distance', x=0.01, ha='left', fontsize=14)
fig.tight_layout()
fig.savefig(f'docs/figures/attention_by_offset{SFX}.png', dpi=style.DPI)

# ---------- 3. logit lens: decode the residual stream after each block ----------
val_text = ''.join(chars[i] for i in val_data[:20000].tolist())
start = val_text.index('\n\n') + 2
passage = val_text[start:start + 48]
idx = torch.tensor([[chars.index(c) for c in passage]])
with torch.no_grad():
    _, _, res = model(idx, return_residuals=True)
    lens = [F.softmax(model.lm_head(model.ln_f(r)), -1)[0] for r in res]  # each (T, vocab)
    # lens loss per depth on the eval set
    lens_loss = []
    _, _, res_all = model(X[:512], return_residuals=True)
    for r in res_all:
        logits = model.lm_head(model.ln_f(r))
        lens_loss.append(F.cross_entropy(logits.view(-1, len(chars)), Y[:512].reshape(-1)).item())
print('logit lens loss by depth:', np.round(lens_loss, 3))

nxt = [chars.index(c) for c in val_text[start + 1:start + 49]]
P = np.array([[lens[d][t, nxt[t]].item() for t in range(48)] for d in range(n_layer + 1)])
top = [[show(chars[lens[d][t].argmax().item()]) for t in range(48)] for d in range(n_layer + 1)]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={'width_ratios': [6, 1]})
ax.imshow(P, cmap='viridis', vmin=0, vmax=1, aspect='auto')
for d in range(n_layer + 1):
    for t in range(48):
        ax.text(t, d, top[d][t], ha='center', va='center', fontsize=9, family='monospace',
                color='#111111' if P[d, t] > 0.5 else '#ffffff')  # viridis runs dark to light in both themes
ax.set_xticks(range(48), [show(c) for c in passage], family='monospace', fontsize=9)
ax.set_yticks(range(n_layer + 1), ['embedding'] + [f'after block {l + 1}' for l in range(n_layer)])
ax.set_title("Logit lens: the model's best guess for the next character at each depth "
             "(color = probability on the true next character)", loc='left')
ax.tick_params(axis='x', length=0)
ax2.barh(range(n_layer + 1), lens_loss, color=style.ACCENTS[0])
ax2.set_ylim(n_layer + 0.5, -0.5)
ax2.set_yticks([])
ax2.set_title('loss', loc='left')
for i, v in enumerate(lens_loss):
    ax2.text(v, i, f' {v:.2f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(f'docs/figures/logit_lens{SFX}.png', dpi=style.DPI)

# ---------- 4. embeddings ----------
def pca2(W):
    W = W - W.mean(0)
    U, S, V = torch.linalg.svd(W, full_matrices=False)
    return (W @ V[:2].T).numpy()

tok = pca2(model.token_embedding_table.weight.detach())
kind = lambda c: 0 if c in 'aeiou' else 1 if c.islower() else 2 if c.isupper() else 3
names = ['lowercase vowel', 'lowercase consonant', 'uppercase', 'space / punctuation / digit']
pos = model.position_embedding_table.weight.detach()
pos = pos / pos.norm(dim=1, keepdim=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4))
for k in range(4):
    sel = [i for i, c in enumerate(chars) if kind(c) == k]
    a1.scatter(tok[sel, 0], tok[sel, 1], s=0, label=names[k])
    for i in sel:
        a1.text(tok[i, 0], tok[i, 1], show(chars[i]), color=style.ACCENTS[k], fontsize=12,
                ha='center', va='center', family='monospace', weight='bold')
a1.legend(handles=[plt.Line2D([], [], ls='', marker='s', color=style.ACCENTS[k], label=names[k]) for k in range(4)],
          frameon=False, fontsize=9, loc='best')
a1.set_title('Character embeddings (first 2 principal components)', loc='left')
a1.set_xticks([]); a1.set_yticks([])
im = a2.imshow((pos @ pos.T).numpy(), cmap=style.SEQ, vmin=-1, vmax=1)
a2.set_title('Position embeddings: cosine similarity', loc='left')
a2.set_xlabel('position'); a2.set_ylabel('position')
fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(f'docs/figures/embeddings{SFX}.png', dpi=style.DPI)

# ---------- 5. training timelapse ----------
log = json.load(open('train_log.json'))
steps = [r['step'] for r in log['losses']]
fig = plt.figure(figsize=(12.8, 7.2))
ax_t = fig.add_axes([0.04, 0.05, 0.56, 0.86]); ax_t.axis('off')
ax_l = fig.add_axes([0.66, 0.15, 0.31, 0.7])
ax_l.plot(steps, [r['train'] for r in log['losses']], color=style.ACCENTS[1], label='train')
ax_l.plot(steps, [r['val'] for r in log['losses']], color=style.ACCENTS[0], label='val')
ax_l.set_xlabel('step'); ax_l.set_ylabel('loss'); ax_l.legend(frameon=False)
ax_l.grid(color=style.GRID)
dot, = ax_l.plot([], [], 'o', color=style.FG)
title = fig.text(0.04, 0.94, '', fontsize=16, weight='bold')
body = ax_t.text(0, 1, '', va='top', family='monospace', fontsize=10.5)


def frame(i):
    s = log['samples'][i]
    v = min(log['losses'], key=lambda r: abs(r['step'] - s['step']))['val']  # nearest eval step
    title.set_text(f"step {s['step']}   ·   val loss {v:.2f}")
    lines = s['text'].strip('\n').split('\n')
    body.set_text('\n'.join(l[:70] for l in lines[:16]))
    dot.set_data([s['step']], [v])
    return title, body, dot


frames = [i for i in range(len(log['samples'])) for _ in range(3)]  # hold each checkpoint ~1.5 s
anim = animation.FuncAnimation(fig, frame, frames=frames + [frames[-1]] * 6)
anim.save(f'docs/figures/timelapse{SFX}.gif', writer=animation.PillowWriter(fps=2), dpi=80)
frame(len(log['samples']) - 1)
fig.savefig(f'docs/figures/timelapse_final{SFX}.png', dpi=style.DPI)

# ---------- 6. data for the explorer page (the page runs the model itself; this is just stats + presets) ----------
passages = []
pos_ = 0
while len(passages) < 8:
    s = val_text.index('\n\n', pos_) + 2
    pos_ = s + 300
    passages.append(val_text[s:s + block_size])
json.dump({'base_loss': round(base, 4), 'heads': heads, 'groups': group_delta,
           'lens_loss': [round(v, 3) for v in lens_loss], 'passages': passages},
          open('docs/data.json', 'w'), separators=(',', ':'))
print('wrote docs/figures/ and docs/data.json')
