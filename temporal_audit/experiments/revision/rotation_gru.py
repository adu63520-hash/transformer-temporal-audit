"""Train three GRU seeds per presentation regime on the paired rotation task."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

def sample(count, rng, device):
    phases = rng.uniform(0, 2 * np.pi, count)
    theta = phases[:, None] + 2 * np.pi * np.arange(24)[None, :] / 24
    positive = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    negative = positive[:, (-np.arange(24)) % 24]
    x = torch.tensor(np.concatenate([positive, negative]), dtype=torch.float32, device=device)
    y = torch.tensor(np.repeat([1, 0], count), dtype=torch.float32, device=device)
    return x, y

def shuffle(x):
    indexes = torch.rand(x.shape[:2], device=x.device).argsort(1)
    return x.gather(1, indexes[..., None].expand_as(x))

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[0]).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    rng = np.random.default_rng(90)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test, y_test = sample(1080, rng, device)
    train, y_train = sample(4096, rng, device)
    torch.manual_seed(7)
    test_shuffles = [shuffle(test) for _ in range(5)]
    results = {}
    for regime in ("ordered", "shuffled"):
        results[regime] = []
        for seed in range(3):
            torch.manual_seed(seed)
            net = GRUClassifier().to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
            for epoch in range(10):
                for idx in torch.randperm(len(train), device=device).split(256):
                    x = train[idx]
                    optimizer.zero_grad()
                    logits = net(x if regime == "ordered" else shuffle(x))
                    nn.functional.binary_cross_entropy_with_logits(logits, y_train[idx]).backward()
                    optimizer.step()
            net.eval()
            with torch.no_grad():
                def accuracy(x):
                    return ((net(x) > 0) == (y_test > 0)).float().mean().item()
                results[regime].append({"seed": seed, "ordered": accuracy(test),
                                        "shuffled": float(np.mean([accuracy(x) for x in test_shuffles]))})
        print(regime, results[regime], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

