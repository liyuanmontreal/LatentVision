# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np
try:
    from sklearn.manifold import trustworthiness as sk_trustworthiness
    HAS_SK_TRUST = True
except Exception:
    HAS_SK_TRUST = False

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X*X, axis=1, keepdims=True)
    D = sq + sq.T - 2 * (X @ X.T)
    np.maximum(D, 0, out=D)
    return np.sqrt(D, out=D)

def kruskal_stress(Dh: np.ndarray, Dl: np.ndarray, weighted: bool=False) -> float:
    if weighted:
        W = 1.0/(Dh + 1e-12)
        num = np.sum(W*(Dh-Dl)**2); den = np.sum(W*(Dh**2))
    else:
        num = np.sum((Dh-Dl)**2); den = np.sum(Dh**2)
    return float(np.sqrt(num/(den+1e-12)))

def _rank_matrix(D: np.ndarray) -> np.ndarray:
    N = D.shape[0]
    D2 = D.copy(); np.fill_diagonal(D2, np.inf)
    order = np.argsort(D2, axis=1)
    ranks = np.empty_like(order)
    for i in range(N):
        ranks[i, order[i]] = np.arange(1, N+1)
    return ranks

def trustworthiness(Xh: np.ndarray, Xl: np.ndarray, k: int=15) -> float:
    if HAS_SK_TRUST:
        return float(sk_trustworthiness(Xh, Xl, n_neighbors=k, metric="euclidean"))
    N = Xh.shape[0]
    Dh = pairwise_distances(Xh); Dl = pairwise_distances(Xl)
    Rh = _rank_matrix(Dh)
    t = 0.0
    for i in range(N):
        low_order = np.argsort(Dl[i]); low_neighbors = [j for j in low_order if j!=i][:k]
        high_order = np.argsort(Dh[i]); high_set = set([j for j in high_order if j!=i][:k])
        for j in low_neighbors:
            if j not in high_set:
                t += (Rh[i,j]-k)
    norm = N*k*(2*N - 3*k - 1)
    return 1.0 - (2.0/norm)*t

def continuity(Xh: np.ndarray, Xl: np.ndarray, k: int=15) -> float:
    N = Xh.shape[0]
    Dh = pairwise_distances(Xh); Dl = pairwise_distances(Xl)
    Rl = _rank_matrix(Dl)
    c = 0.0
    for i in range(N):
        high_order = np.argsort(Dh[i]); high_neighbors = [j for j in high_order if j!=i][:k]
        low_order = np.argsort(Dl[i]); low_set = set([j for j in low_order if j!=i][:k])
        for j in high_neighbors:
            if j not in low_set:
                c += (Rl[i,j]-k)
    norm = N*k*(2*N - 3*k - 1)
    return 1.0 - (2.0/norm)*c

def lcmc(Xh: np.ndarray, Xl: np.ndarray, k: int=15) -> float:
    N = Xh.shape[0]
    Dh = pairwise_distances(Xh); Dl = pairwise_distances(Xl)
    overlap = 0.0
    for i in range(N):
        Nh = set([j for j in np.argsort(Dh[i]) if j!=i][:k])
        Nl = set([j for j in np.argsort(Dl[i]) if j!=i][:k])
        overlap += len(Nh & Nl)/float(k)
    overlap /= N
    baseline = k/float(N-1)
    return float(overlap - baseline)

def triplet_preservation(Xh: np.ndarray, Xl: np.ndarray, num_triplets: int=50000, seed: int=42) -> float:
    rng = np.random.default_rng(seed)
    N = Xh.shape[0]
    Dh = pairwise_distances(Xh); Dl = pairwise_distances(Xl)
    count = 0; correct = 0
    for _ in range(num_triplets):
        i, j, l = rng.integers(0, N, size=3)
        if i==j or i==l or j==l: continue
        dij, dil = Dh[i,j], Dh[i,l]
        if dij==dil: continue
        d2ij, d2il = Dl[i,j], Dl[i,l]
        correct += int((d2ij<d2il) == (dij<dil))
        count += 1
    return float('nan') if count==0 else float(correct)/float(count)

def evaluate_all(Xh: np.ndarray, Xl: np.ndarray, k: int=15, weighted_stress: bool=False,
                 triplets: int=50000, seed: int=42) -> dict:
    Dh = pairwise_distances(Xh); Dl = pairwise_distances(Xl)
    return {
        "stress": kruskal_stress(Dh, Dl, weighted=False),
        "stress_weighted": kruskal_stress(Dh, Dl, weighted=weighted_stress),
        f"trustworthiness@{k}": trustworthiness(Xh, Xl, k=k),
        f"continuity@{k}": continuity(Xh, Xl, k=k),
        f"lcmc@{k}": lcmc(Xh, Xl, k=k),
        "triplet_preservation": triplet_preservation(Xh, Xl, num_triplets=triplets, seed=seed),
    }
