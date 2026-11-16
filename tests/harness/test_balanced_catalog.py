from collections import Counter
import pytest
import yaml
from chisel.balanced_catalog import BalancedCatalog


def manifest(tmp_path):
    # Hash ownership would assign every checkpoint to rank zero.
    entries = [dict(domain=domain, path=f'{domain}{i}.pt', sha256=f'{(i + offset) * 8:064x}')
               for domain, offset in [('cifar', 0), ('text', 16)] for i in range(8)]
    path = tmp_path / 'manifest.yaml'
    path.write_text(yaml.safe_dump(dict(states=entries)))
    return path


def test_balanced_ownership_epochs_and_resume(tmp_path):
    path = manifest(tmp_path)
    catalog = BalancedCatalog(path, dict(cifar=4, text=4), seed=7, owner_count=4)
    batches = [next(catalog) for _ in range(2)]
    for batch in batches:
        counts = Counter((entry['domain'], catalog.owner(entry['sha256'], 4)) for entry in batch)
        assert counts == {(domain, rank): 1 for domain in ('cifar', 'text') for rank in range(4)}
    assert len({entry['sha256'] for batch in batches for entry in batch}) == 16
    assert all(catalog.owner(entry['sha256'], 1) == 0 for entry in batches[0])
    next(catalog)
    saved = catalog.state_dict()
    resumed = BalancedCatalog(path, dict(cifar=4, text=4), seed=99, owner_count=4)
    resumed.load_state_dict(saved)
    for _ in range(8):
        assert next(catalog) == next(resumed)


def test_rejects_uneven_or_changed_topology(tmp_path):
    path = manifest(tmp_path)
    with pytest.raises(ValueError, match='divisible'):
        BalancedCatalog(path, dict(cifar=3, text=4), seed=7, owner_count=4)
    catalog = BalancedCatalog(path, dict(cifar=4, text=4), seed=7, owner_count=4)
    with pytest.raises(ValueError, match='world size'):
        catalog.owner('0' * 64, 2)
    other = BalancedCatalog(path, dict(cifar=4, text=4), seed=7, owner_count=2)
    with pytest.raises(ValueError, match='topology'):
        other.load_state_dict(catalog.state_dict())
