"""
train_tf.py -- end-to-end GPT-style regressor for beta, plus the permutation ruler.

  mode : orig | rand          task variant (see tasks.py)
  cond : clean | time | full  training-time intervention applied to EVERY training sequence
                              (clean = none; time = hourly triples shuffled; full = 72 values permuted)
At evaluation every model is scored on real / time-shuffled / fully-permuted versions of the same
ID (interior grid) and OOD (border ring) test sets.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tasks as T


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if hasattr(F, "scaled_dot_product_attention"):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            w = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
            a = torch.softmax(w.masked_fill(mask, float("-inf")), -1) @ v
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class GPTRegressor(nn.Module):
    """Decoder-only (causal) Transformer; scalar read out from the last position."""

    def __init__(self, d=256, layers=4, heads=4, embed="token+value"):
        super().__init__()
        self.embed = embed
        self.tok = nn.Embedding(T.VOCAB_SIZE, d)
        self.pos = nn.Parameter(torch.zeros(1, T.SEQ_LEN, d))
        self.val = nn.Linear(1, d) if embed == "token+value" else None
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.tok.weight, std=0.02); nn.init.normal_(self.pos, std=0.02)

    def forward(self, tok):
        x = self.tok(tok) + self.pos
        if self.val is not None:
            is_val = (tok >= T.TEMP_OFFSET).float()
            v = (((tok - T.TEMP_OFFSET).float() * T.DELTA + T.V_MIN) - 20.0) / 10.0 * is_val
            x = x + self.val(v.unsqueeze(-1)) * is_val.unsqueeze(-1)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, -1])).squeeze(-1)


@torch.no_grad()
def predict(model, bins, mu, sd, device, bs=1024):
    out = []
    for i in range(0, len(bins), bs):
        tok = torch.from_numpy(T.to_tokens(bins[i:i + bs])).to(device)
        out.append(model(tok).float().cpu().numpy())
    return np.exp(np.clip(np.concatenate(out) * sd + mu, -12, 2))


def evaluate(model, tests, mu, sd, device, n_perm, keep_preds=False):
    model.eval()
    res = {}
    for split, d in tests.items():
        r = {}
        p = predict(model, d["bins"], mu, sd, device)
        r["real"] = T.metrics(p, d["beta"])
        if keep_preds:
            r["pred_real"] = [float(x) for x in p]
        for kind in ("time", "full"):
            rng = np.random.default_rng(7)          # same permutations for every model
            ms = [T.metrics(predict(model, T.intervene(d["bins"], kind, rng), mu, sd, device), d["beta"])
                  for _ in range(n_perm)]
            r[kind] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            r["R_" + kind] = r[kind]["mre"] / max(r["real"]["mre"], 1e-9)
            r["Rlog_" + kind] = r[kind]["mae_log"] / max(r["real"]["mae_log"], 1e-9)
        res[split] = r
    model.train()
    return res


def run(mode, cond, seed, out_path, steps=6000, bs=256, n_train=20000, d=256, layers=4, heads=4,
        lr=5e-4, embed="token+value", eval_every=1000, reps=30, n_perm=5, log=print, ckpt_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(10_000 + seed)

    train = T.make_dataset(mode, "train", n_train, seed=1000)          # fixed benchmark training set
    tests = {"id": T.make_dataset(mode, "id", reps, seed=2000), "ood": T.make_dataset(mode, "ood", reps, seed=3000)}
    logb = np.log(train["beta"]); mu, sd = float(logb.mean()), float(logb.std())
    z = ((logb - mu) / sd).astype(np.float32)

    log(f"  data ready ({n_train} train), building model on {device} ...")
    model = GPTRegressor(d, layers, heads, embed).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    curve, t0, run_loss = [], time.time(), None
    model.train()
    for step in range(1, steps + 1):
        idx = rng.integers(0, n_train, bs)
        bins = T.intervene(train["bins"][idx], cond if cond != "clean" else "real", rng)
        tok = torch.from_numpy(T.to_tokens(bins)).to(device)
        y = torch.from_numpy(z[idx]).to(device)
        loss = F.mse_loss(model(tok), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1:
            log(f"  first step done, loss {loss.item():.4f}, {time.time() - t0:.1f}s")
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % eval_every == 0 or step == steps:
            ev = evaluate(model, tests, mu, sd, device, n_perm=2 if step < steps else n_perm)
            c = {"step": step, "train_loss": run_loss, "sec": time.time() - t0}
            for s in ("id", "ood"):
                c[s + "_mre"] = ev[s]["real"]["mre"]; c[s + "_medre"] = ev[s]["real"]["medre"]
                c[s + "_R_time"] = ev[s]["R_time"]; c[s + "_R_full"] = ev[s]["R_full"]
                c[s + "_mre_time"] = ev[s]["time"]["mre"]; c[s + "_mre_full"] = ev[s]["full"]["mre"]
            curve.append(c)
            log(f"  [{mode}/{cond}/s{seed}] step {step:5d} loss {run_loss:.4f} | ID MRE {c['id_mre']:.3f} "
                f"(med {c['id_medre']:.3f}) R_time {c['id_R_time']:.2f} R_full {c['id_R_full']:.1f} | "
                f"OOD MRE {c['ood_mre']:.3f} R_time {c['ood_R_time']:.2f} | {step / (time.time() - t0):.1f} it/s")

    final = evaluate(model, tests, mu, sd, device, n_perm=n_perm, keep_preds=True)
    result = {"mode": mode, "cond": cond, "seed": seed, "embed": embed, "steps": steps, "bs": bs,
              "n_train": n_train, "d": d, "layers": layers, "heads": heads, "lr": lr, "n_params": n_params,
              "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "torch": torch.__version__, "train_sec": time.time() - t0, "curve": curve, "final": final}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if ckpt_path is not None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "d": d, "layers": layers,
                    "heads": heads, "embed": embed}, ckpt_path)
    return result
