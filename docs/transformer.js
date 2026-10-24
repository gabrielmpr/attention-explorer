// The same forward pass as model.py, in plain JavaScript, so the page can run the model live.
// Linear weights are stored (out, in) row-major, like nn.Linear.
const GPT = (() => {

  function load(manifest, buffer) {
    const all = new Float32Array(buffer);
    const W = {};
    for (const [name, { shape, offset }] of Object.entries(manifest.tensors)) {
      W[name] = all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    }
    return { ...manifest, W, V: manifest.chars.length, stoi: Object.fromEntries(manifest.chars.map((c, i) => [c, i])) };
  }

  // x: (T, inD) -> (T, outD)
  function linear(x, T, inD, w, b, outD) {
    const y = new Float32Array(T * outD);
    for (let t = 0; t < T; t++) {
      const xo = t * inD, yo = t * outD;
      for (let o = 0; o < outD; o++) {
        let s = b ? b[o] : 0;
        const wo = o * inD;
        for (let i = 0; i < inD; i++) s += x[xo + i] * w[wo + i];
        y[yo + o] = s;
      }
    }
    return y;
  }

  function layernorm(x, T, D, g, b) {
    const y = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let mean = 0, v = 0;
      for (let i = 0; i < D; i++) mean += x[o + i];
      mean /= D;
      for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2;
      const inv = 1 / Math.sqrt(v / D + 1e-5);
      for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mean) * inv * g[i] + b[i];
    }
    return y;
  }

  function softmaxRows(x, T, D) {
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let m = -Infinity, s = 0;
      for (let i = 0; i < D; i++) m = Math.max(m, x[o + i]);
      for (let i = 0; i < D; i++) { x[o + i] = Math.exp(x[o + i] - m); s += x[o + i]; }
      for (let i = 0; i < D; i++) x[o + i] /= s;
    }
    return x;
  }

  // ids: int array (T <= block_size). mask[l][h] = 0 switches a head off.
  // Returns probs (T, V), att[l][h] (T, T), and lens[d] (T, V) = the prediction read off after d blocks.
  function forward(M, ids, mask, { lens = true } = {}) {
    const { W, n_embd: C, n_head: H, n_layer: L, V } = M, T = ids.length, hs = C / H, scale = 1 / Math.sqrt(hs);
    let x = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < C; i++) x[t * C + i] = W['token_embedding_table.weight'][ids[t] * C + i] + W['position_embedding_table.weight'][t * C + i];
    }
    const resid = [x.slice()], att = [];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`;
      const h = layernorm(x, T, C, W[p + 'ln1.weight'], W[p + 'ln1.bias']);
      const cat = new Float32Array(T * C);
      att.push([]);
      for (let hh = 0; hh < H; hh++) {
        const q = linear(h, T, C, W[`${p}sa.heads.${hh}.query.weight`], null, hs);
        const k = linear(h, T, C, W[`${p}sa.heads.${hh}.key.weight`], null, hs);
        const v = linear(h, T, C, W[`${p}sa.heads.${hh}.value.weight`], null, hs);
        const a = new Float32Array(T * T);
        for (let t = 0; t < T; t++) {
          let m = -Infinity;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            for (let d = 0; d < hs; d++) s += q[t * hs + d] * k[j * hs + d];
            a[t * T + j] = s * scale;
            m = Math.max(m, a[t * T + j]);
          }
          let z = 0;
          for (let j = 0; j <= t; j++) { a[t * T + j] = Math.exp(a[t * T + j] - m); z += a[t * T + j]; }
          for (let j = 0; j <= t; j++) a[t * T + j] /= z;
        }
        att[l].push(a);
        const on = mask[l][hh];
        if (!on) continue;
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < hs; d++) {
            let s = 0;
            for (let j = 0; j <= t; j++) s += a[t * T + j] * v[j * hs + d];
            cat[t * C + hh * hs + d] = s * on;
          }
        }
      }
      const sa = linear(cat, T, C, W[p + 'sa.proj.weight'], W[p + 'sa.proj.bias'], C);
      for (let i = 0; i < x.length; i++) x[i] += sa[i];
      const h2 = layernorm(x, T, C, W[p + 'ln2.weight'], W[p + 'ln2.bias']);
      const f = linear(h2, T, C, W[p + 'ffwd.net.0.weight'], W[p + 'ffwd.net.0.bias'], 4 * C);
      for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
      const f2 = linear(f, T, 4 * C, W[p + 'ffwd.net.2.weight'], W[p + 'ffwd.net.2.bias'], C);
      for (let i = 0; i < x.length; i++) x[i] += f2[i];
      resid.push(x.slice());
    }
    const head = r => softmaxRows(linear(layernorm(r, T, C, W['ln_f.weight'], W['ln_f.bias']), T, C, W['lm_head.weight'], W['lm_head.bias'], V), T, V);
    return { T, probs: head(x), att, lens: lens ? resid.map(head) : null };
  }

  function encode(M, s) { return [...s].filter(c => c in M.stoi).map(c => M.stoi[c]); }

  return { load, forward, encode };
})();

if (typeof module !== 'undefined') module.exports = GPT;
