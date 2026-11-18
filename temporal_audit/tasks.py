"""
tasks.py  -- data generation, tokenization, interventions, analytic anchors (numpy only).

Two task variants built on the same two-stage relaxation chain:
    T1[t] = T1[t-1] + beta * (F[t] - T1[t-1])
    T2[t] = T2[t-1] + beta * (T1[t-1] - T2[t-1])
  "orig": the paper's setting  -- fixed diurnal sinusoid forcing, fixed initial state (20, 20).
  "rand": shortcut-reduced     -- per-sample random forcing spectrum and random initial state,
                                  so that amplitude / level statistics no longer identify beta.

Token layout is identical to data/tokenizer_shared.py (5 header tokens + 24 x 6 hourly tokens).
"""
import numpy as np

N_TIME = 24
DT = 3600.0
DELTA = 0.1          # temperature quantisation step (TEMP_SPEC)
V_MIN, V_MAX = -50.0, 100.0
TEMP_OFFSET = 7 + 501 + 501   # specials(2)+struct(5)+len bins+alpha bins, as in PDEVocabulary
VOCAB_SIZE = TEMP_OFFSET + 1501
SEQ_LEN = 5 + 6 * N_TIME      # 149, no padding
PAD, BOS, P_LEN, P_ALPHA, H_OUT, H_IN, H_ZONE = 0, 1, 2, 3, 4, 5, 6

LEN_GRID = np.linspace(0.5, 2.0, 8)
LOGA_GRID = np.linspace(-6.0, -4.7, 8)
INNER = list(range(1, 7))


def beta_of(length, log_alpha):
    return (10.0 ** log_alpha) * DT / (length ** 2)


# ----------------------------------------------------------------------------- parameters
def sample_interior(n, rng):
    """Continuous sampling inside the interior 6x6 box (training distribution)."""
    L = rng.uniform(LEN_GRID[1], LEN_GRID[6], n)
    la = rng.uniform(LOGA_GRID[1], LOGA_GRID[6], n)
    return L, la


def grid_params(ood, reps):
    """Grid points of the original benchmark: interior 36 (ID) or border ring 28 (OOD)."""
    L, la = [], []
    for li in range(8):
        for ai in range(8):
            inner = (li in INNER) and (ai in INNER)
            if inner == ood:
                continue
            L += [LEN_GRID[li]] * reps
            la += [LOGA_GRID[ai]] * reps
    return np.array(L), np.array(la)


# ----------------------------------------------------------------------------- forcing
def make_forcing(n, mode, rng):
    t = np.arange(N_TIME)[None, :]
    if mode == "orig":
        return 20.0 + 8.0 * np.sin(2 * np.pi * t / N_TIME - np.pi / 2) + rng.normal(0, 0.5, (n, N_TIME))
    if mode == "rand":
        base = rng.uniform(12.0, 28.0, (n, 1))
        amp = rng.uniform(0.0, 8.0, (n, 1))
        freq = rng.uniform(0.5, 4.0, (n, 1))
        phase = rng.uniform(0, 2 * np.pi, (n, 1))
        rho = rng.uniform(0.0, 0.95, (n, 1))
        scale = rng.uniform(0.5, 5.0, (n, 1))
        eps = rng.normal(0, 1, (n, N_TIME))
        ar = np.zeros((n, N_TIME))
        ar[:, 0] = eps[:, 0]
        for k in range(1, N_TIME):
            ar[:, k] = rho[:, 0] * ar[:, k - 1] + np.sqrt(1 - rho[:, 0] ** 2) * eps[:, k]
        return base + amp * np.sin(2 * np.pi * freq * t / N_TIME + phase) + scale * ar
    raise ValueError(mode)


def simulate(beta, forcing, mode, rng):
    """Vectorised rollout. Returns series [n, 24, 3] = (F, T1, T2), un-quantised."""
    n = len(beta)
    if mode == "orig":
        T1 = np.full(n, 20.0); T2 = np.full(n, 20.0)
    else:
        T1 = rng.uniform(10.0, 30.0, n); T2 = rng.uniform(10.0, 30.0, n)
    out = np.zeros((n, N_TIME, 3))
    for t in range(N_TIME):
        T1n = T1 + beta * (forcing[:, t] - T1)
        T2n = T2 + beta * (T1 - T2)
        T1, T2 = T1n, T2n
        out[:, t, 0] = forcing[:, t]; out[:, t, 1] = T1; out[:, t, 2] = T2
    return out


def quantise(series):
    return np.round((np.clip(series, V_MIN, V_MAX) - V_MIN) / DELTA).astype(np.int64)   # bin index


def dequantise(bins):
    return V_MIN + bins * DELTA


def make_dataset(mode, split, n_or_reps, seed):
    """split: 'train' (n continuous interior samples) | 'id' | 'ood' (grid points x reps).
    Returns dict(bins [n,24,3] int64, beta [n])."""
    rng = np.random.default_rng(seed)
    if split == "train":
        L, la = sample_interior(n_or_reps, rng)
    else:
        L, la = grid_params(ood=(split == "ood"), reps=n_or_reps)
    beta = beta_of(L, la)
    series = simulate(beta, make_forcing(len(beta), mode, rng), mode, rng)
    return {"bins": quantise(series), "beta": beta}


def make_period_dataset(period, reps, seed):
    """Cue-conflict test set: the fixed-forcing task with the forcing period changed (training uses 24 h).
    Trajectories remain valid rollouts of the same system, so the recurrence (and OLS) is unaffected,
    while amplitude statistics now map to a different beta. Interior grid points x reps."""
    rng = np.random.default_rng(seed)
    L, la = grid_params(ood=False, reps=reps)
    beta = beta_of(L, la)
    t = np.arange(N_TIME)[None, :]
    forcing = 20.0 + 8.0 * np.sin(2 * np.pi * t / period - np.pi / 2) + rng.normal(0, 0.5, (len(beta), N_TIME))
    return {"bins": quantise(simulate(beta, forcing, "orig", rng)), "beta": beta}


# ----------------------------------------------------------------------------- interventions
def intervene(bins, kind, rng):
    """bins [n,24,3].  kind:
       'real'  -- untouched
       'time'  -- the 24 hourly triples are shuffled jointly (channel identity and simultaneity
                  preserved; only temporal order destroyed)
       'full'  -- all 72 value tokens permuted (the paper's content permutation: destroys temporal
                  order AND channel identity)"""
    if kind == "real":
        return bins
    n = bins.shape[0]
    out = bins.copy()
    if kind == "time":
        idx = np.argsort(rng.random((n, N_TIME)), axis=1)
        return np.take_along_axis(out, idx[:, :, None], axis=1)
    if kind == "full":
        flat = out.reshape(n, 72)
        idx = np.argsort(rng.random((n, 72)), axis=1)
        return np.take_along_axis(flat, idx, axis=1).reshape(n, N_TIME, 3)
    raise ValueError(kind)


def to_tokens(bins):
    """Serialise to the benchmark token layout with header VALUE tokens masked (-> PAD)."""
    n = bins.shape[0]
    tok = np.zeros((n, SEQ_LEN), dtype=np.int64)
    tok[:, 0] = BOS; tok[:, 1] = P_LEN; tok[:, 2] = PAD; tok[:, 3] = P_ALPHA; tok[:, 4] = PAD
    for t in range(N_TIME):
        b = 5 + 6 * t
        tok[:, b] = H_OUT;      tok[:, b + 1] = TEMP_OFFSET + bins[:, t, 0]
        tok[:, b + 2] = H_IN;   tok[:, b + 3] = TEMP_OFFSET + bins[:, t, 1]
        tok[:, b + 4] = H_ZONE; tok[:, b + 5] = TEMP_OFFSET + bins[:, t, 2]
    return tok


# ----------------------------------------------------------------------------- analytic anchors
def ols_beta(bins):
    """Closed-form OLS on de-quantised tokens, using only the 23 within-sequence transitions
    (no knowledge of the initial state needed)."""
    x = dequantise(bins)
    F, T1, T2 = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    X1 = F[:, 1:] - T1[:, :-1]; d1 = T1[:, 1:] - T1[:, :-1]
    X2 = T1[:, :-1] - T2[:, :-1]; d2 = T2[:, 1:] - T2[:, :-1]
    num = (X1 * d1).sum(1) + (X2 * d2).sum(1)
    den = (X1 ** 2).sum(1) + (X2 ** 2).sum(1)
    return num / np.maximum(den, 1e-12)


_Q = [0, .05, .1, .25, .5, .75, .9, .95, 1]


def _stats(v):
    q = np.quantile(v, _Q, axis=1).T
    return np.concatenate([q, v.mean(1, keepdims=True), v.std(1, keepdims=True)], axis=1)


def feat_pooled_bag(bins):
    """Invariant to the FULL permutation: statistics of the unordered multiset of 72 values."""
    return _stats(dequantise(bins).reshape(len(bins), 72))


def feat_triple_bag(bins):
    """Invariant to the TIME shuffle: statistics of the unordered set of 24 (F,T1,T2) triples."""
    x = dequantise(bins)
    F, T1, T2 = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    blocks = [_stats(F), _stats(T1), _stats(T2), _stats(F - T1), _stats(T1 - T2), _stats(F - T2)]

    def corr(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        return ((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12))[:, None]
    blocks += [corr(F, T1), corr(T1, T2), corr(F, T2)]
    return np.concatenate(blocks, axis=1)


class RFFRidge:
    """Ridge regression on standardised features + random Fourier features (a cheap nonlinear,
    training-free-ish order-blind predictor). Target: log beta."""

    def __init__(self, n_rff=1500, l2=1e-1, gamma=0.5, seed=0):
        self.n_rff, self.l2, self.gamma, self.seed = n_rff, l2, gamma, seed

    def _phi(self, A):
        Z = (A - self.m) / self.s
        return np.concatenate([Z, np.cos(Z @ self.W + self.b), np.ones((len(Z), 1))], axis=1)

    def fit(self, A, logy):
        rng = np.random.default_rng(self.seed)
        self.m, self.s = A.mean(0), A.std(0) + 1e-8
        self.W = rng.normal(0, self.gamma / np.sqrt(A.shape[1]), (A.shape[1], self.n_rff))
        self.b = rng.uniform(0, 2 * np.pi, self.n_rff)
        P = self._phi(A)
        self.w = np.linalg.solve(P.T @ P + self.l2 * np.eye(P.shape[1]), P.T @ logy)
        return self

    def predict(self, A):
        return np.exp(np.clip(self._phi(A) @ self.w, -12, 2))


# ----------------------------------------------------------------------------- metrics
def _rank(a):
    r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(len(a)); return r


def metrics(pred, beta):
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 1e-6)
    lp, lb = np.log(pred), np.log(beta)
    re = np.abs(pred - beta) / beta
    return {
        "mre": float(re.mean()), "medre": float(np.median(re)),
        "mae_log": float(np.abs(lp - lb).mean()),
        "pearson_log": float(np.corrcoef(lp, lb)[0, 1]) if lp.std() > 0 else 0.0,
        "spearman": float(np.corrcoef(_rank(pred), _rank(beta))[0, 1]) if lp.std() > 0 else 0.0,
        "r2_log": float(1 - ((lp - lb) ** 2).sum() / ((lb - lb.mean()) ** 2).sum()),
    }
