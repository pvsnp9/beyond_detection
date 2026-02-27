
import numpy as np


def _apply_top_k(mask: np.ndarray, sim: np.ndarray, top_k: int | None) -> np.ndarray:
    if top_k is None or top_k <= 0:
        return mask

    kept = np.zeros_like(mask, dtype=bool)
    for i in range(mask.shape[0]):
        cols = np.where(mask[i])[0]
        if cols.size == 0:
            continue
        if cols.size <= top_k:
            kept[i, cols] = True
            continue
        order = np.argsort(sim[i, cols])[::-1]
        top_cols = cols[order[:top_k]]
        kept[i, top_cols] = True
    return kept


def _greedy_match(sim: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    edges: list[tuple[float, int, int]] = []
    for i in range(mask.shape[0]):
        js = np.where(mask[i])[0]
        for j in js:
            edges.append((float(sim[i, j]), i, int(j)))
    edges.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_i: set[int] = set()
    used_j: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, i, j in edges:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matched.append((i, j))
    return matched


def _assignment_match(sim: np.ndarray, mask: np.ndarray, mode: str) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # noqa: BLE001
        return _greedy_match(sim, mask)

    n_pred, n_ref = sim.shape
    if n_pred == 0 or n_ref == 0:
        return []

    n_total = n_pred + n_ref
    weights = np.zeros((n_total, n_total), dtype=np.float64)

    if mode == "max_cardinality":
        real_weights = np.where(mask, 1.0 + (1e-3 * sim), -1e6)
    else:
        real_weights = np.where(mask, sim, -1e6)

    weights[:n_pred, :n_ref] = real_weights

    rows, cols = linear_sum_assignment(-weights)

    pairs: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < n_pred and j < n_ref and mask[i, j]:
            pairs.append((int(i), int(j)))
    return pairs


def one_to_one_match(
    S: np.ndarray,
    tau: float,
    mode: str,
    top_k: int | None,
) -> list[tuple[int, int]]:
    """Compute one-to-one matches between rows (pred) and cols (ref)."""
    if S.ndim != 2:
        raise ValueError("S must be a 2D similarity matrix")
    if S.size == 0:
        return []

    sim = np.asarray(S, dtype=np.float32)
    mask = sim >= float(tau)
    mask = _apply_top_k(mask=mask, sim=sim, top_k=top_k)

    if mode == "greedy":
        return _greedy_match(sim, mask)
    if mode == "max_cardinality":
        return _assignment_match(sim, mask, mode="max_cardinality")
    if mode == "max_weight":
        return _assignment_match(sim, mask, mode="max_weight")

    return _greedy_match(sim, mask)
