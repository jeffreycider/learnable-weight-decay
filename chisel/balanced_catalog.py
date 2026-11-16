"""Equal per-domain checkpoint work on every private-probe owner."""
from chisel.offline_data import Catalog


class BalancedCatalog(Catalog):
    def __init__(self, manifest, batch_counts, seed, owner_count):
        super().__init__(manifest, batch_counts, seed)
        if owner_count < 1:
            raise ValueError('owner_count must be positive')
        self.owner_count, self.owners = owner_count, {}
        shards, counts = {}, {}
        for domain, entries in self.domains.items():
            if len(entries) % owner_count or batch_counts[domain] % owner_count:
                raise ValueError('domain size and batch count must be divisible by owner_count')
            ordered = sorted(entries, key=lambda entry: entry['sha256'])
            for rank in range(owner_count):
                key = (domain, rank)
                shards[key] = ordered[rank::owner_count]
                counts[key] = batch_counts[domain] // owner_count
                self.owners.update((entry['sha256'], rank) for entry in shards[key])
        self.domains, self.batch_counts = shards, counts

    def owner(self, identity, world_size):
        if world_size == 1:
            return 0  # Allows exact single-process reference execution of this sampler.
        if world_size != self.owner_count:
            raise ValueError('checkpoint owner_count must match distributed world size')
        # Independent audit catalogs may include checkpoints absent from the fitting set.
        return self.owners.get(identity, int(identity, 16) % world_size)

    def state_dict(self):
        return super().state_dict() | dict(owner_count=self.owner_count)

    def load_state_dict(self, saved):
        if saved.get('owner_count') != self.owner_count:
            raise ValueError('checkpoint owner topology changed')
        super().load_state_dict(saved)
