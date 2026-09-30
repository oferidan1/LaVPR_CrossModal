"""
Pairwise McNemar over saved probe hits - no GPU, no re-run.

The probe tested every estimator against `random`, but the claim that matters
is "attention beats IDF", and "A beats random" plus "B does not beat random"
is NOT evidence that A beats B. This runs every pairing directly.

    python probe_pairwise.py logs/.../per_query_hits.npz [--k 10]
    python probe_pairwise.py runA.npz runB.npz --k 10     # two models at once

A LOWER hit rate after deletion means the estimator found more important
tokens, so the winner of each pairing is the one with fewer hits.
"""

import sys
import argparse
import numpy as np


def mcnemar(a, b):
    """Paired exact test. Returns (n_a_only, n_b_only, p)."""
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    try:
        from scipy.stats import binomtest
        return n01, n10, float(binomtest(n01, n, 0.5).pvalue)
    except ImportError:
        from math import erfc, sqrt
        z = (abs(n01 - n10) - 1) / sqrt(n)
        return n01, n10, float(erfc(z / sqrt(2)))


def stars(p, n_tests=1):
    """Bonferroni-adjusted, so a table of pairings is not read as a table of
    independent findings."""
    q = min(1.0, p * n_tests)
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else '  '


def load(path, k):
    d = np.load(path)
    base = d['none']
    names, hits = [], {}
    for key in d.files:
        if key == 'none' or not key.endswith(f'_k{k}'):
            continue
        n = key[:-len(f'_k{k}')]
        names.append(n)
        hits[n] = d[key]
    return base, names, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='+')
    ap.add_argument('--k', type=int, default=10)
    a = ap.parse_args()

    for path in a.npz:
        base, names, hits = load(path, a.k)
        n = len(base)
        print(f"\n=== {path}  k={a.k}  {n} queries ===")
        print(f"baseline R@1 {100.0 * base.mean():.2f}\n")

        order = sorted(names, key=lambda x: hits[x].mean())
        print(f"{'estimator':<13} dR@1")
        for x in order:
            print(f"{x:<13} {100.0 * (base.mean() - hits[x].mean()):6.2f}")

        n_tests = len(order) * (len(order) - 1) // 2
        print(f"\npairwise McNemar, Bonferroni over {n_tests} pairings")
        print(f"{'':<13}" + "".join(f"{x[:10]:>12}" for x in order))
        for i, x in enumerate(order):
            row = f"{x:<13}"
            for j, y in enumerate(order):
                if i == j:
                    row += f"{'-':>12}"
                elif j < i:
                    row += f"{'':>12}"
                else:
                    _, _, p = mcnemar(hits[x], hits[y])
                    row += f"{p:>9.4f}{stars(p, n_tests)}"
            print(row)
        print("\nrows are ordered most-destructive first, so an entry that is "
              "significant\nmeans the ROW estimator beat the COLUMN estimator.")


if __name__ == '__main__':
    main()
