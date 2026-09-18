import numpy as np

from ecup_matching.splits import connected_component_ids, stratified_component_folds


def test_connected_components_join_transitive_product_pairs() -> None:
    components = connected_component_ids([1, 2, 10], [2, 3, 11])

    assert components[0] == components[1]
    assert components[0] != components[2]


def test_component_folds_never_split_a_group() -> None:
    targets = np.tile([0, 1], 20)
    categories = np.repeat(["a", "b"], 20)
    groups = np.arange(40)
    groups[1::2] = groups[::2]

    folds = stratified_component_folds(
        targets,
        categories,
        groups,
        n_splits=2,
        random_state=2026,
    )

    for group in np.unique(groups):
        assert np.unique(folds[groups == group]).size == 1
