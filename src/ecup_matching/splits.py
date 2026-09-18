"""Leakage-safe fold generation for pairwise product matching."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._size: dict[int, int] = {}

    def find(self, item: int) -> int:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1
            return item

        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != item:
            next_item = self._parent[item]
            self._parent[item] = root
            item = next_item
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._size[left_root] < self._size[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size.pop(right_root)


def connected_component_ids(id1: Iterable[int], id2: Iterable[int]) -> NDArray[np.int64]:
    """Return a stable component identifier for every pair.

    Components are built over every labeled edge, regardless of its target. Keeping a
    complete component in one fold guarantees that product IDs cannot leak across folds.
    """

    left = np.asarray(list(id1), dtype=np.int64)
    right = np.asarray(list(id2), dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("id1 and id2 must have equal lengths")

    union_find = _UnionFind()
    for left_id, right_id in zip(left, right, strict=True):
        union_find.union(int(left_id), int(right_id))

    roots = np.fromiter(
        (union_find.find(int(item)) for item in left),
        dtype=np.int64,
        count=len(left),
    )
    unique_roots = np.unique(roots)
    stable_ids = np.searchsorted(unique_roots, roots).astype(np.int64, copy=False)
    return stable_ids


def stratified_component_folds(
    targets: Iterable[int | float],
    categories: Iterable[str],
    component_ids: Iterable[int],
    *,
    n_splits: int = 5,
    random_state: int = 2026,
) -> NDArray[np.int8]:
    """Assign whole components to category-and-label-stratified folds."""

    target = np.asarray(list(targets), dtype=np.int8)
    category = np.asarray(list(categories), dtype=object)
    groups = np.asarray(list(component_ids), dtype=np.int64)
    if not (len(target) == len(category) == len(groups)):
        raise ValueError("targets, categories, and component_ids must have equal lengths")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("targets must be binary")

    strata = np.asarray(
        [
            f"{category_name}\x1f{label}"
            for category_name, label in zip(category, target, strict=True)
        ],
        dtype=object,
    )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = np.full(len(target), -1, dtype=np.int8)
    placeholder = np.zeros((len(target), 1), dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(placeholder, strata, groups=groups)
    ):
        folds[validation_indices] = fold

    if (folds < 0).any():
        raise RuntimeError("not every row received a validation fold")
    return folds
