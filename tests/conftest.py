import pytest

@pytest.fixture
def dummy_partition_manifest():
    return {
        "dataset": "dummy",
        "dataset_split": "train",
        "strategy": "dirichlet",
        "alpha": 0.1,
        "n_clients": 10,
        "seed": 42,
        "n_train_samples": 1000,
        "partition_format_version": 1,
    }
