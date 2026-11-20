"""E3 -- watershed proposal generation (spec E3, step 4).

anchors = gated atoms union atoms with z >= 0.5 surprisal. Valley atoms
(non-anchors) drain to their nearest left anchor; proposals are the
maximal contiguous valley runs between anchors, ordered by estimated cost
(sum of atom surprisal) ascending -- cheapest, most-confidently-foldable
spans get committed first.
"""
import numpy as np

Z_ANCHOR_THRESHOLD = 0.5


def _zscore(values: list[float]) -> list[float]:
    arr = np.array(values)
    std = arr.std()
    if std == 0:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / std).tolist()


def build_proposals(atoms: list[dict], atom_surprisals: list[float]):
    z = _zscore(atom_surprisals)
    is_anchor = [atoms[i]["gated"] or z[i] >= Z_ANCHOR_THRESHOLD for i in range(len(atoms))]

    proposals = []
    n = len(atoms)
    i = 0
    while i < n:
        if is_anchor[i]:
            i += 1
            continue
        start = i
        while i < n and not is_anchor[i]:
            i += 1
        end = i  # exclusive
        cost = float(sum(atom_surprisals[start:end]))
        proposals.append({"atom_start": start, "atom_end": end, "cost": cost})

    proposals.sort(key=lambda p: p["cost"])
    return proposals, is_anchor, z
