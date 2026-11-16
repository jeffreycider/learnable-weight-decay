"""Replayable CIFAR batches: dataset identity + YAML + step determine every draw."""
import hashlib
from functools import lru_cache
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@lru_cache(maxsize=4)
def load_dataset(path, sha256, size, modified):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError('CIFAR dataset SHA256 mismatch')
    return torch.load(path, map_location='cpu', weights_only=True)


def seed_at(seed, index):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{index}'.encode(), digest_size=8).digest(), 'little') % (2**63 - 1)


def reference_policy(**kwargs):
    # Pinned third-party code uses NumPy's removed integer alias during construction.
    from vendor.cifar_recipe.utils.autoaug import CIFAR10Policy
    old = getattr(np, 'int', None)
    try:
        np.int = int
        return CIFAR10Policy(**kwargs)
    finally:
        if old is None:
            del np.int
        else:
            np.int = old


class Images(Dataset):
    def __init__(self, data, transform):
        self.data, self.transform = data, transform

    def __getitem__(self, key):
        from PIL import Image
        index, seed = key
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                image = self.transform(Image.fromarray(self.data['images'][index].numpy()))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return image, self.data['labels'][index]


def stratified_indices(labels, split_seed, heldout_per_class, split):
    if split not in ('train', 'validation', 'all'):
        raise ValueError('unknown CIFAR split')
    generator = torch.Generator().manual_seed(split_seed)
    selected = []
    for label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        selected.append(indices if split == 'all' else indices[:heldout_per_class] if split == 'validation' else indices[heldout_per_class:])
    return torch.cat(selected)


class CifarStream:
    def __init__(self, path, sha256, split, split_seed, heldout_per_class, batch_size,
                 repeats, minibatch_seed, augmentation_seed, transform, mix, workers,
                 device, rank=0, world_size=1):
        info = Path(path).stat()
        data = load_dataset(str(Path(path).resolve()), sha256, info.st_size, info.st_mtime_ns)
        self.dataset = Images(data, transform)
        self.indices = stratified_indices(data['labels'], split_seed, heldout_per_class, split)
        self.batch_size, self.repeats = batch_size, repeats
        if batch_size % world_size or len(self.indices) * repeats < batch_size:
            raise ValueError('invalid global CIFAR batch')
        self.minibatch_seed, self.augmentation_seed = minibatch_seed, augmentation_seed
        self.device, self.rank, self.world_size = device, rank, world_size
        self.mix, self.step, self.iterator = mix, 0, None
        self.cached_epoch, self.order = None, None
        self.loader = DataLoader(self.dataset, batch_sampler=self, num_workers=workers,
                                 pin_memory=str(device).startswith('cuda'),
                                 generator=torch.Generator().manual_seed(augmentation_seed))

    def recipe(self, step):
        usable = len(self.indices) * self.repeats // self.batch_size
        epoch, within = divmod(step, usable)
        if self.cached_epoch != epoch:
            orders = []
            for repeat in range(self.repeats):
                generator = torch.Generator().manual_seed(seed_at(self.minibatch_seed, epoch * self.repeats + repeat))
                orders.append(self.indices[torch.randperm(len(self.indices), generator=generator)])
            self.cached_epoch, self.order = epoch, torch.cat(orders)
        start = within * self.batch_size + self.rank * (self.batch_size // self.world_size)
        ids = self.order[start:start + self.batch_size // self.world_size].tolist()
        draw = epoch * len(self.indices) * self.repeats + start
        return [(index, seed_at(self.augmentation_seed, draw + offset)) for offset, index in enumerate(ids)]

    def __iter__(self):
        # DataLoader may prefetch ahead; its cursor must not advance the consumed step.
        step = self.step
        while True:
            yield self.recipe(step)
            step += 1

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        inputs, labels = next(self.iterator)
        return self.finish(inputs, labels)

    def finish(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs, targets = self.mix(inputs, labels, self.step) if self.mix else (inputs, labels)
        batch = dict(inputs=inputs, targets=targets, recipe=dict(step=self.step, kind='cifar'))
        self.step += 1
        return batch

    def replay(self, step):
        examples = [self.dataset[key] for key in self.recipe(step)]
        inputs, labels = (torch.stack(items).to(self.device) for items in zip(*examples))
        inputs, targets = self.mix(inputs, labels, step) if self.mix else (inputs, labels)
        return dict(inputs=inputs, targets=targets)

    def state_dict(self):
        return dict(step=self.step)

    def preview(self):
        """Next augmented batch without advancing or replacing prefetch workers."""
        return self.replay(self.step)

    def load_state_dict(self, state):
        self.step, self.iterator = state['step'], None


class Mix:
    def __init__(self, seed, probability, cutmix_probability, alpha, classes):
        self.seed, self.probability, self.cutmix_probability = seed, probability, cutmix_probability
        self.alpha, self.classes = alpha, classes

    def __call__(self, inputs, labels, step):
        rng = np.random.RandomState(seed_at(self.seed, step) % 2**32)
        targets = torch.nn.functional.one_hot(labels, self.classes).float()
        if rng.rand() >= self.probability:
            return inputs, targets
        permutation = torch.as_tensor(rng.permutation(len(inputs)), device=inputs.device)
        fraction = rng.beta(self.alpha, self.alpha)
        if rng.rand() < self.cutmix_probability:
            height, width = inputs.shape[-2:]
            cut_h, cut_w = int(height * np.sqrt(1 - fraction)), int(width * np.sqrt(1 - fraction))
            y, x = rng.randint(height), rng.randint(width)
            y0, y1 = max(0, y - cut_h // 2), min(height, y + cut_h // 2)
            x0, x1 = max(0, x - cut_w // 2), min(width, x + cut_w // 2)
            inputs = inputs.clone()
            inputs[:, :, y0:y1, x0:x1] = inputs[permutation, :, y0:y1, x0:x1]
            fraction = 1 - (y1 - y0) * (x1 - x0) / (height * width)
        else:
            inputs = inputs * fraction + inputs[permutation] * (1 - fraction)
        return inputs, targets * fraction + targets[permutation] * (1 - fraction)
